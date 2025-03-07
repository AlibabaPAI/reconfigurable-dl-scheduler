import asyncio
import collections
import copy
import dateutil.parser
import jsonpatch
import json
import kubernetes_asyncio as kubernetes
import logging
import math
import k8s_templates as templates
import config as config
import time
from datetime import datetime, timezone
from prometheus_client import Counter, Summary
import collections
from resources import set_default_resources
from utils import patch_job_status
import os

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class JobsController(object):
    """
    The main controller responsible for the overall DLJobs lifecycle.
    Essentially, it keeps a queue of DLJobs whose states may need to be
    synchronized. It watches for events such as pod status changes and
    allocation changes and enqueues any potentially affects DLJobs. A
    worker coroutine is responsible for processing DLJobs from the queue
    and guarantees that a single DLJob is never processed concurrently.
    """

    def __init__(self):
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("morphling.alibaba.com", "v1", "", "morphlingjobs")
        self._queue = asyncio.Queue()

    async def run(self):
        # Create service if it doesn't already exist.
        # FIXME: initialize allocations
        await asyncio.gather(
            self._watch_jobs(), self._watch_pods(), self._sync_worker()
        )

    async def _watch_jobs(self):
        # Watch for changes to DLJobs and enqueue them to be synced.
        # Perform a full reconcile after every timeout.
        async with kubernetes.watch.Watch() as watch:
            while True:
                async for event in watch.stream(
                    self._objs_api.list_namespaced_custom_object,
                    *self._custom_resource,
                    timeout_seconds=10,
                ):
                    job_name = event["object"]["metadata"]["name"]
                    namespace = event["object"]["metadata"]["namespace"]
                    await self._queue.put((namespace, job_name))

    async def _watch_pods(self):
        # Watch for changes to pods and enqueue their DLJobs to be synced.
        # Perform a full reconcile after every timeout.
        async with kubernetes.watch.Watch() as watch:
            while True:
                async for event in watch.stream(
                    self._core_api.list_namespaced_pod,
                    "",
                    label_selector="morphling/job",
                    timeout_seconds=60,
                ):
                    pod = event["object"]
                    job_name = pod.metadata.labels["morphling/job"]
                    namespace = pod.metadata.namespace
                    await self._queue.put((namespace, job_name))

    async def _sync_worker(self):
        while True:
            (namespace, name) = await self._queue.get()
            await self._sync_job(namespace, name)
            self._queue.task_done()

    async def _sync_job(self, namespace, job_name):
        current_ts = datetime.now(timezone.utc)
        job, pyJob, pods = await self._get_job_and_pyJob(namespace, job_name)
        if job is not None:
            job = await self._validate_pyJob(job, pyJob, pods)
        if job is None:  # Not Found, presumably was deleted.
            remainingSteps = await self._delete_pyJob(pyJob, pods)
            return
        # Use ChainMap to record updates to the job status fields.
        job["status"] = collections.ChainMap({}, job.get("status", {}))
        if job["status"] == {} or "remainingSteps" not in job["status"]:
            job["status"]["remainingSteps"] = job["spec"]["steps"]
        if "placement" in job["spec"]:
            job["status"]["allocation"] = job["spec"]["placement"]
        # Get the current phase of the job, None if no phase was set.
        allocation = job["status"].get("allocation", {})
        cpuID = job["status"].get("cpu_alloc", {})
        phase = job["status"].setdefault("phase", "Pending")
        replicas = job["status"].get("replicas", 0)
        if completion_status := self._detect_completion(
            pyJob, pods, job["status"]["remainingSteps"]
        ):
            # Job is already completed.
            job["status"].update(completion_status)
            job["status"].setdefault("completionTimestamp", current_ts)
            job["status"]["allocation"] = allocation = {}
            remainingSteps = await self._delete_pyJob(pyJob, pods)
            job["status"]["remainingSteps"] = remainingSteps
        elif phase == "Pending":
            if (
                "nvidia.com/gpu" in allocation
                and allocation["nvidia.com/gpu"]
                and not pods
            ):
                # Start the next group of pods.
                job["status"]["phase"] = "Starting"
        elif phase == "Starting":
            # FIXME: In case if a pod experiences indefinite ImagePullBackOff,
            # the job can get stuck in the Starting phase
            if (
                (
                    self._count_scheduled_pods(pods) != replicas
                    and self._detect_restart(pyJob, pods, copy.deepcopy(allocation))
                )
                or not allocation
                or not allocation["nvidia.com/gpu"]
            ):
                # Allocator changed allocation based on new information about
                # resource availability
                job["status"]["phase"] = "Stopping"
            elif allocation and not pyJob:
                # Start the next group of pods.
                job["status"]["group"] = job["status"].get("group", -1) + 1
                await self._create_pytorch_job(
                    job["metadata"],
                    job["spec"]["application"],
                    job["spec"]["template"],
                    copy.deepcopy(allocation),
                    job["status"]["group"],
                    cpuID,
                    job["spec"]["localBsz"],
                    job["spec"]["workDir"],
                    job["status"]["remainingSteps"],
                )
                job["status"]["lease_time"] = time.time()
            elif (
                self._count_gpus(pods) != replicas
                and (
                    datetime.now()
                    - datetime.strptime(
                        pyJob["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    )
                ).total_seconds()
                > 90
            ):
                # Controller restarted before we can spawn all pods
                job["status"]["phase"] = "Stopping"
            elif self._count_ready_pods(pods) == replicas:
                # All pods are running
                job["status"]["phase"] = "Running"
        elif phase == "Running":
            if self._detect_restart(pyJob, pods, copy.deepcopy(allocation)) or not pods:
                # 1. Reallocation OR 2. the controller restarted before
                # we can update phase to Stopping
                job["status"]["phase"] = "Stopping"
        elif phase == "Stopping":
            if pods:
                remainingSteps = await self._delete_pyJob(
                    pyJob, pods, job["status"]["remainingSteps"]
                )
                job["status"]["remainingSteps"] = remainingSteps
            else:
                # All pods successfully deleted
                job["status"]["phase"] = "Pending"
        # Set replicas and ready replicas.
        if allocation:
            job["status"]["replicas"] = len(allocation["nvidia.com/gpu"])
            job["status"]["readyReplicas"] = self._count_ready_pods(pods)
        else:
            job["status"]["allocation"] = None
            job["status"]["replicas"] = None
            job["status"]["readyReplicas"] = None
        # Update attained service if needed.
        attained_service = job["status"].setdefault("attainedService", 0)
        attained_service_ts = job["status"].setdefault(
            "attainedServiceTimestamp", current_ts
        )
        if isinstance(attained_service_ts, str):
            attained_service_ts = dateutil.parser.isoparse(attained_service_ts)
        if replicas != (job["status"].get("replicas") or 0):
            duration = (current_ts - attained_service_ts).total_seconds()
            job["status"]["attainedService"] += duration * replicas
            job["status"]["attainedServiceTimestamp"] = current_ts
        # Apply changes to MorphlingJob status.
        patch = {
            "status": {
                k: v
                for k, v in job["status"].maps[0].items()
                if v != job["status"].maps[1].get(k)
            }
        }
        if patch["status"]:
            LOG.info("Patch MorphlingJob %s: %s", job_name, patch)
            await patch_job_status(self._objs_api, namespace, job_name, patch)

    def _count_ready_pods(self, pods):
        count = 0
        for pod in pods:
            if pod.status.container_statuses and all(
                stat.ready for stat in pod.status.container_statuses
            ):
                count += int(pod.spec.containers[0].resources.limits["nvidia.com/gpu"])
        return count

    def _count_scheduled_pods(self, pods):
        count = 0
        for pod in pods:
            if pod.status.conditions and any(
                cond.type == "PodScheduled" and cond.status == "True"
                for cond in pod.status.conditions
            ):
                count += int(pod.spec.containers[0].resources.limits["nvidia.com/gpu"])
        return count

    def _count_gpus(self, pods):
        count = 0
        for pod in pods:
            count += int(pod.spec.containers[0].resources.limits["nvidia.com/gpu"])
        return count

    async def _get_job_and_pyJob(self, namespace, name):
        try:
            job = await self._objs_api.get_namespaced_custom_object(
                "morphling.alibaba.com", "v1", namespace, "morphlingjobs", name
            )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                return None, None, None
            raise  # Unexpected error.
        try:
            pyJob = await self._objs_api.list_namespaced_custom_object(
                "kubeflow.org",
                "v1",
                namespace,
                "pytorchjobs",
                label_selector=f"morphling/job={name}",
            )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                return job, None, None
            raise  # Unexpected error.
        if len(pyJob["items"]) > 0:
            pyJob = pyJob["items"][0]
            pyJobName = pyJob["metadata"]["name"]
            pods = await self._core_api.list_namespaced_pod(
                namespace, label_selector=f"training.kubeflow.org/job-name={pyJobName}"
            )
            pods = pods.items
        else:
            pyJob = None
            pods = []
        return job, pyJob, pods

    async def _validate_pyJob(self, job, pyJob, pods):
        namespace = job["metadata"]["namespace"]
        name = job["metadata"]["name"]
        patch_status = {}
        # Validate pods for job.
        if pyJob is None:
            return job
        try:
            alloc = json.loads(pyJob["metadata"]["annotations"]["allocation"])
        except (KeyError, ValueError):
            tmp_name = pyJob["metadata"]["name"]
            patch_status["phase"] = "Failed"
            patch_status["reason"] = "Invalid"
            patch_status["message"] = f"invalid annotations for pyJob {tmp_name}"
        gpuAlloc = alloc["Master"] + alloc["Worker"]
        for pod in pods:
            # Check the pod is running on the correct node.
            if pod.spec.node_name and pod.spec.node_name not in gpuAlloc:
                patch_status["phase"] = "Failed"
                patch_status["reason"] = "Invalid"
                patch_status["message"] = f"incorrect node for pod {pod.metadata.name}"
                break
            if pod.spec.node_name and pod.spec.node_name in gpuAlloc:
                gpuAlloc.remove(pod.spec.node_name)
        if patch_status:
            return await patch_job_status(
                self._objs_api, namespace, name, {"status": patch_status}
            )
        return job

    def _detect_completion(self, pyJob, pods, remaining_steps):
        if not pyJob or len(pods) == 0:
            return {}
        gpuAlloc = json.loads(pyJob["metadata"]["annotations"]["allocation"])
        replicas = len(gpuAlloc["Master"] + gpuAlloc["Worker"])

        # Check if all pods succeeded.
        for pod in pods:
            if len(pods) != replicas or pod.status.phase != "Succeeded":
                break
        else:
            return {"phase": "Succeeded"}

        if os.path.exists("/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log"):
            file = open("/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log")
            try:
                lines = file.readlines()
                last_line = lines[-1]
            except:
                LOG.info(
                    "Can not read lines %s",
                    "/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log",
                )
                last_line = 1
            if int(last_line) >= int(remaining_steps):
                file.close()
                return {"phase": "Succeeded"}

        for pod in pods:
            if pod.status.phase == "Unknown":
                # This can happen if there's something wrong with the node
                # or kubelet assigned to this pod.
                LOG.warning("Unknown status for pod %s", pod.metadata.name)
            elif pod.status.phase != "Failed":
                continue
            if pod.status.reason == "UnexpectedAdmissionError":
                # This can happen if a non-DL pod claims the node's
                # resources before this pod could bind to that node.
                LOG.warning(
                    "UnexpectedAdmissionError for pod %s: %s",
                    pod.metadata.name,
                    pod.status.message,
                )
            elif str(pod.status.reason).startswith("Outof"):
                # we might be temporarily out of pods on this node
                LOG.warning(
                    f"Pod {pod.metadata.name} is {pod.status.reason} "
                    f"on {pod.spec.node_name}"
                )
            else:
                return {
                    "phase": "Failed",
                    "reason": "PodFailure",
                    "message": f"{pod.metadata.name} {pod.status.phase}",
                }
        return {}

    def _detect_restart(self, pyJob, pods, allocation):
        gpuAlloc = allocation["nvidia.com/gpu"]
        if (
            self._count_gpus(pods) != len(gpuAlloc)
            and len(pods) != 0
            and (
                datetime.now()
                - datetime.strptime(
                    pyJob["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                )
            ).total_seconds()
            > 90
        ):
            return True
        for pod in pods:
            if pod.spec.node_name:
                n = pod.spec.node_name
                for i in range(
                    0, int(pod.spec.containers[0].resources.limits["nvidia.com/gpu"])
                ):
                    if n in gpuAlloc:
                        gpuAlloc.remove(n)
                    else:
                        return True
        return False

    async def _delete_pyJob(self, pyJob, pods, remaining_steps=0):
        if pyJob is None:
            return remaining_steps
        for pod in pods:
            if pod.status.phase == "Failed":
                return remaining_steps
        results, names = [], []
        if "deletion_timestamp" not in pyJob["metadata"]:
            results.append(
                self._objs_api.delete_namespaced_custom_object(
                    group="kubeflow.org",
                    version="v1",
                    namespace=pyJob["metadata"]["namespace"],
                    plural="pytorchjobs",
                    name=pyJob["metadata"]["name"],
                    body=kubernetes.client.V1DeleteOptions(),
                )
            )
            names.append(pyJob["metadata"]["name"])
        if results:
            LOG.info(f"Deleting {names}")
            await asyncio.gather(*results, return_exceptions=True)
            if remaining_steps != 0 and os.path.exists(
                "/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log"
            ):
                file = open("/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log")
                try:
                    lines = file.readlines()
                    last_line = lines[-1]
                except:
                    LOG.info(
                        "Can not read lines %s",
                        "/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log",
                    )
                    last_line = 1
                remaining_steps -= int(last_line)
                file.close()
        return remaining_steps

    async def _create_pytorch_job(
        self,
        job_metadata,
        application,
        pod_template,
        allocation,
        group,
        cpuIDs,
        local_bsz,
        workDir,
        steps,
    ):
        obj_args = ("kubeflow.org", "v1", "default", "pytorchjobs")
        pyJob = {}
        pyJob["apiVersion"] = "kubeflow.org/v1"
        pyJob["kind"] = "PyTorchJob"
        pyJob.setdefault("metadata", {})
        pyJob["metadata"]["name"] = self._get_pyjob_name(job_metadata, group)
        pyJob["metadata"]["ownerReferences"] = templates.owner_reference_template(
            job_metadata["namespace"],
            job_metadata["name"],
            job_metadata["uid"],
            kind="MorphlingJob",
        )
        pyJob["metadata"].setdefault("labels", {})
        pyJob["metadata"]["labels"]["morphling/job"] = job_metadata["name"]
        pyJob["metadata"].setdefault("annotations", {})
        nodeList = []
        for n in allocation["nvidia.com/gpu"]:
            node = await self._core_api.read_node(n)
            nodeList.append(node.metadata.labels["kubernetes.io/hostname"])
        if len(nodeList) == 0:
            return
        nodeCount = collections.Counter(nodeList)
        nproc_per_node = gcd_many([v for k, v in nodeCount.items()])
        num_pods = len(nodeList) // nproc_per_node

        gpuAlloc = {}
        isMaster = True
        gpuAlloc.setdefault("Worker", [])
        for k, v in nodeCount.items():
            while v:
                if isMaster:
                    gpuAlloc["Master"] = [k]
                    isMaster = False
                else:
                    gpuAlloc["Worker"].append(k)
                v -= nproc_per_node
        pyJob["metadata"]["annotations"]["allocation"] = json.dumps(gpuAlloc)
        if allocation["strategy"][2] == "zero-offload" and cpuIDs != {}:
            masterNode = gpuAlloc["Master"][0]
            cpuMasterId = []
            for i in range(0, nproc_per_node):
                cpuMasterId += cpuIDs[masterNode].pop(0)
            pyJob["metadata"]["annotations"]["cpuMasterID"] = json.dumps(
                {masterNode: [cpuMasterId]}
            )
            if len(cpuIDs[masterNode]) == 0:
                del cpuIDs[nodeList[0]]
            cpuWorkerAlloc = {}
            for k, v in cpuIDs.items():
                if k in nodeList:
                    while len(v) != 0:
                        cpuWorkerID = []
                        for i in range(0, nproc_per_node):
                            cpuWorkerID += v.pop(0)
                        cpuWorkerAlloc.setdefault(k, [])
                        cpuWorkerAlloc[k].append(cpuWorkerID)

            pyJob["metadata"]["annotations"]["cpuWorkerID"] = json.dumps(cpuWorkerAlloc)
        pyJob.setdefault("spec", {})
        pyJob["spec"].setdefault("pytorchReplicaSpecs", {})
        pyJob["spec"]["pytorchReplicaSpecs"].setdefault("Master", {})
        pyJob["spec"]["pytorchReplicaSpecs"].setdefault("Worker", {})
        pyJob["spec"]["pytorchReplicaSpecs"]["Master"]["replicas"] = 1
        pyJob["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = num_pods - 1
        pyJob["spec"]["pytorchReplicaSpecs"]["Master"]["restartPolicy"] = pyJob["spec"][
            "pytorchReplicaSpecs"
        ]["Worker"]["restartPolicy"] = "OnFailure"
        micro_bsz = local_bsz // (num_pods * nproc_per_node)
        for idx, container in enumerate(pod_template["spec"]["containers"]):
            container.setdefault("env", [])
            container["env"].append(
                {
                    "name": "NNODES",
                    "value": str(num_pods),
                }
            )
            container["env"].append(
                {
                    "name": "STEPS",
                    "value": str(steps),
                }
            )
            container["env"].append(
                {
                    "name": "NPROC",
                    "value": str(nproc_per_node),
                }
            )
            container["env"].append(
                {
                    "name": "LOGGING_DIR",
                    "value": "/mnt/nasdata/ckpt/" + pyJob["metadata"]["name"] + ".log",
                }
            )
            container["env"].append(
                {
                    "name": "NGPUS",
                    "value": str(len(allocation["nvidia.com/gpu"])),
                }
            )
            if application in ["vit", "roberta", "swin"]:
                if allocation["strategy"][2] == "zero-offload":
                    container["args"][0] += (
                        "  --deepspeed " + workDir + "config/zero_opt.json "
                    )
                    container["env"].append(
                        {
                            "name": "OMP_NUM_THREADS",
                            "value": str(int(len(allocation["cpu"]) / len(nodeList))),
                        }
                    )
                elif allocation["strategy"][2] == "zero-dp":
                    container["args"][0] += (
                        "  --deepspeed " + workDir + "config/zero_2_opt.json "
                    )
                elif allocation["strategy"][1] == "gc":
                    container["args"][0] += "  --gradient_checkpointing true "
            else:
                if allocation["strategy"][0] == "3D":
                    container["args"] = container["args"][:1]
                    micro_bsz = str(len(allocation["nvidia.com/gpu"]))
                    if len(allocation["nvidia.com/gpu"]) == 2 and application in [
                        "bert",
                        "gpt2",
                    ]:
                        micro_bsz = 1
                elif allocation["strategy"][1] == "gc":
                    container["args"] = container["args"][:1]
                    container["args"][0] += "  --checkpoint-activations "
                    micro_bsz = local_bsz
                else:
                    micro_bsz = 1
                    if allocation["strategy"][2] == "zero-offload":
                        container["args"] = [container["args"][1]]
                    elif allocation["strategy"][2] == "zero-dp":
                        container["args"] = container["args"][2:]
            container["env"].append(
                {
                    "name": "BATCH_SIZE",
                    "value": str(micro_bsz),
                }
            )
            container["resources"]["limits"]["nvidia.com/gpu"] = nproc_per_node
            container["resources"]["requests"]["nvidia.com/gpu"] = nproc_per_node
        pyJob["spec"]["pytorchReplicaSpecs"]["Master"]["template"] = pyJob["spec"][
            "pytorchReplicaSpecs"
        ]["Worker"]["template"] = pod_template
        if pyJob["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] == 0:
            del pyJob["spec"]["pytorchReplicaSpecs"]["Worker"]
        await self._objs_api.create_namespaced_custom_object(*obj_args, pyJob)

    def _patch_pods_and_containers(self, pod):
        pod_patch = config.get_job_patch_pods()
        container_patch = config.get_job_patch_containers()
        if pod_patch:
            pod = jsonpatch.apply_patch(pod, pod_patch)
        if container_patch:
            for idx, container in enumerate(pod["spec"]["containers"]):
                pod["spec"]["containers"][idx] = jsonpatch.apply_patch(
                    container, container_patch
                )
        return pod

    def _get_pyjob_name(self, job_metadata, group):
        job_name = job_metadata["name"]
        job_uid = job_metadata["uid"]
        return f"{job_name}-{job_uid}-{group}"


def gcd_many(s):
    g = 0
    for i in range(len(s)):
        if i == 0:
            g = s[i]
        else:
            g = math.gcd(g, s[i])

    return g


def list_as_string(list_of_int):
    str_list = [str(int_val) for int_val in list_of_int]
    return str_list
