import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
from datetime import datetime, timezone
import collections

from applications import APPLICATIONS
from model import MODELS
from morphling_sched.policy.synergy import SynergyPolicy
from morphling_sched.policy.antman import antmanPolicy
from job_info import JobInfo, NodeInfo
from resources import (
    get_node_configuration,
)
from utils import patch_job_status, replace_job_status
from config import allowed_taints

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


POLICY = "morphling-r"
assert POLICY in [
    "morphling",
    "morphling-n",
    "pollux",
    "synergy",
    "antman",
    "morphling-r",
]

if POLICY == "synergy":
    RSC_TYPE = ["nvidia.com/gpu", "cpu", "memory"]
elif POLICY == "antman" or POLICY == "morphling-n":
    RSC_TYPE = ["nvidia.com/gpu"]
else:
    RSC_TYPE = ["nvidia.com/gpu", "strategy"]
# node configuration
NODE_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

# for synergy
LEASE_TIME = 1800
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


class MorphlingAllocator(object):
    def __init__(self):
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self.rsc_metrix = {}
        self.orig_tpt = {}
        self.max_throughput_config = {}
        self.cpu_available = {}
        self.gpu_available = {}
        LOG.info("policy: %s", POLICY)
        if POLICY == "pollux":
            pass  # self._policy = PolluxPolicy()
        elif POLICY == "synergy":
            self._policy = SynergyPolicy()
        elif POLICY == "antman":
            self._policy = antmanPolicy()

    async def run(self):
        while True:
            LOG.info("Running allocator loop")
            nodes = await self._find_nodes()
            LOG.info("Node resources: %s", {k: v.resources for k, v in nodes.items()})
            jobs, prev_allocations = await self._find_jobs_and_allocations()
            LOG.info("allocator gpu available %s", self.gpu_available)
            LOG.info("allocator gpu prev_allocations %s", prev_allocations)
            start = time.time()
            allocations = self._allocate(jobs, nodes, prev_allocations)
            duration = time.time() - start
            LOG.info("Allocations (in %.3f sec): %s", duration, allocations)
            await self._update_allocations(allocations)
            LOG.info("Sleep for 10 seconds")
            await asyncio.sleep(10)

    async def _update_allocations(self, allocations):
        job_list = await self._objs_api.list_namespaced_custom_object(
            "morphling.alibaba.com", "v1", "", "morphlingjobs"
        )
        for job in job_list["items"]:
            namespace = job["metadata"]["namespace"]
            name = job["metadata"]["name"]
            job_allocation = job.get("status", {}).get("allocation", {})
            new_allocation = {}
            for r in RSC_TYPE:
                new_allocation[r] = allocations[r].get((namespace, name), [])
                if job_allocation == {}:
                    job_allocation.setdefault(r, {})
            job_cpu_alloc = {}
            job_GPU_alloc = job.get("status", {}).get("gpu_alloc", {})
            should_restart = False
            LOG.info("Allocations  % s %s %s", name, job_allocation, new_allocation)
            if job_allocation != new_allocation:
                # check GPU
                gpu_cnt = collections.Counter(new_allocation["nvidia.com/gpu"])
                prev_gpu_cnt = collections.Counter(job_allocation["nvidia.com/gpu"])
                if prev_gpu_cnt != gpu_cnt:
                    ok, job_GPU_alloc = await self._update_gpu_alloc(
                        name, gpu_cnt, prev_gpu_cnt
                    )
                    if not ok:
                        continue
                    should_restart = True

                # check CPU
                cpu_cnt = collections.Counter(new_allocation["cpu"])
                prev_cpu_cnt = collections.Counter(job_allocation["cpu"])
                if prev_cpu_cnt != cpu_cnt:
                    if len(new_allocation["nvidia.com/gpu"]) != 0:
                        prev_thread = len(new_allocation["cpu"]) // len(
                            new_allocation["nvidia.com/gpu"]
                        )
                    else:
                        prev_thread = 0
                    if len(job_allocation["nvidia.com/gpu"]) != 0:
                        now_thread = len(job_allocation["cpu"]) // len(
                            job_allocation["nvidia.com/gpu"]
                        )
                    else:
                        now_thread = 0

                    if (
                        prev_cpu_cnt.keys() != cpu_cnt.keys()
                        or prev_thread != now_thread
                    ):
                        job_cpu_alloc = await self._update_cpu_alloc(
                            name, cpu_cnt, prev_cpu_cnt, now_thread, gpu_cnt
                        )
                        should_restart = True
                placement = []
                alloc = new_allocation["nvidia.com/gpu"]
                for i in range(len(new_allocation["nvidia.com/gpu"])):
                    if i == 0 or alloc[i] != alloc[i - 1]:
                        placement.append(1)
                    else:
                        placement[-1] += 1
                LOG.info("Placements  % s %s ", name, placement)

                if POLICY == "morphling-r":
                    if len(placement) == 0:
                        iteration_time = 0
                    else:
                        placement = tuple(filter(None, placement))
                        iteration_time = APPLICATIONS[
                            job["spec"]["application"]
                        ].get_throughput(placement, new_allocation["strategy"])[2]
                if POLICY == "antman" or POLICY == "morphling-n" or POLICY == "synergy":
                    if len(placement) == 0:
                        iteration_time = 0
                    else:
                        placement = tuple(filter(None, placement))
                        iteration_time = APPLICATIONS[
                            job["spec"]["application"]
                        ].get_throughput(placement, job["spec"]["strategy"])[2]

                patch = {
                    "status": {
                        "allocation": new_allocation,  # allocation
                        "should_restart": should_restart,
                        "iteration_time": iteration_time,
                        "lease_time": datetime.now(),
                        # "gpu_alloc" : job_GPU_alloc,
                    }
                }
                # if job_cpu_alloc:
                # new_status["status"]["cpu_alloc"]=job_cpu_alloc

                LOG.info("Patch MorphlingJob %s/%s: %s", namespace, name, patch)
                await patch_job_status(self._objs_api, namespace, name, patch)
                if job_GPU_alloc:
                    patch = {
                        "status": {
                            "gpu_alloc": job_GPU_alloc,
                        }
                    }
                    LOG.info(
                        "Patch MorphlingJob gpu_alloc %s/%s: %s", namespace, name, patch
                    )
                    await patch_job_status(self._objs_api, namespace, name, patch)

    async def _update_cpu_alloc(self, jobName, cpu_cnt, prev_cpu_cnt, threads, gpu_cnt):
        job_cpu_alloc = {}
        # clear
        for node in prev_cpu_cnt:
            for i in range(0, len(self.cpu_available[node])):
                if self.cpu_available[node][i] == jobName:
                    self.cpu_available[node][i] = ""
        # fulfill
        for node in cpu_cnt:
            alloc_num = 0
            cpu_alloc_num = threads * gpu_cnt[node]
            job_cpu_alloc.setdefault(node, [])
            for i in range(0, len(self.cpu_available[node])):
                if cpu_alloc_num == alloc_num:
                    break
                if self.cpu_available[node][i] == "":
                    self.cpu_available[node][i] = jobName
                    alloc_num += 1
                    job_cpu_alloc[node].append(i * 2)
                # assert not(i==len(self.cpu_available[node])-1 and cpu_alloc_num!=alloc_num)
        return job_cpu_alloc

    async def _update_gpu_alloc(self, jobName, gpu_cnt, prev_gpu_cnt):
        job_GPU_alloc = {}
        for node in NODE_LIST:
            job_GPU_alloc.setdefault(node, [])
        # clear
        for node in prev_gpu_cnt:
            for i in range(0, len(self.gpu_available[node])):
                if self.gpu_available[node][i] == jobName:
                    self.gpu_available[node][i] = ""
        # fulfill
        for node in gpu_cnt:
            gpu_alloc_num = gpu_cnt[node]
            alloc_num = 0
            for i in range(0, len(self.gpu_available[node])):
                if gpu_alloc_num == alloc_num:
                    break
                if self.gpu_available[node][i] == "":
                    self.gpu_available[node][i] = jobName
                    alloc_num += 1
                    job_GPU_alloc[node].append(i)
                if (
                    i == len(self.gpu_available[node]) - 1
                    and gpu_alloc_num != alloc_num
                ):
                    return False, None
        return True, job_GPU_alloc

    async def _find_nodes(self):
        node_infos = {}
        node_list = NODE_LIST
        # Find all non-Morphling pods which are taking up resources and subtract
        # those resources from the available pool. Apparently there's not a
        # more efficient way to get currently available resources in k8s?. We
        # also check if we have reached the pod limit on the node. This number
        # denotes (allocatable pods - Non-terminated pods) on that node.
        # pod_list = await self._core_api.list_pod_for_all_namespaces(
        #         label_selector="!training.kubeflow.org/job-name")
        for node in node_list:
            # if allowed_taints(node.spec.taints):
            resources = get_node_configuration()
            self.cpu_available[node] = [""] * resources["cpu"]
            self.gpu_available[node] = [""] * resources["nvidia.com/gpu"]
            # if not resources.get("pods"):
            #     LOG.warning(f"node {node.metadata.name} "
            #                 "has no free pods available.")
            node_infos[node] = NodeInfo(resources, False)
        return node_infos

    async def _find_jobs_and_allocations(self):
        job_list = await self._objs_api.list_namespaced_custom_object(
            "morphling.alibaba.com", "v1", "", "morphlingjobs"
        )
        job_infos = {}
        allocations = {}
        for r in RSC_TYPE:
            allocations.setdefault(r, {})
        for job in job_list["items"]:
            if job.get("status", {}).get("phase") not in [
                "Pending",
                "Running",
                "Starting",
                "Stopping",
            ]:
                continue
            if "placement" in job["spec"]:
                continue
            if "allocation" in job.get("status", {}):
                namespace = job["metadata"]["namespace"]
                name = job["metadata"]["name"]
                for r in RSC_TYPE:
                    allocations[r][namespace, name] = job["status"]["allocation"][r]
                alloc = allocations["nvidia.com/gpu"].get((namespace, name), [])
                if len(set(alloc)) > 1:
                    job["status"]["is_distributed"] = True
                if "gpu_alloc" in job.get("status", {}) and len(alloc) > 0:
                    namespace = job["metadata"]["namespace"]
                    name = job["metadata"]["name"]
                    job_gpu_alloc = job["status"]["gpu_alloc"]
                    for node in job_gpu_alloc:
                        for gpu_id in job_gpu_alloc[node]:
                            self.gpu_available[int(node)][gpu_id] = name

            name = job["metadata"]["name"]
            namespace = job["metadata"]["namespace"]
            if (
                (
                    POLICY == "morphling-N"
                    or POLICY == "morphling"
                    or POLICY == "morphling-r"
                )
                and name not in self.rsc_metrix
                or name not in self.max_throughput_config
            ):
                # initial resource matrix
                jobModel = MODELS[job["spec"]["application"]]
                if (
                    jobModel.jobSolver.rsc_metrix == {}
                    or jobModel.jobSolver.max_throughput_config == None
                ):
                    jobModel.jobSolver.perfCurve(job["spec"]["application"])
                self.rsc_metrix[name] = jobModel.jobSolver.rsc_metrix
                self.max_throughput_config[name] = (
                    jobModel.jobSolver.max_throughput_config
                )

            if POLICY == "synergy":
                if "lease_time" in job["status"]:
                    lease_time = (
                        datetime.now()
                        - datetime.strptime(job["status"]["lease_time"], TIME_FORMAT)
                    ).total_seconds()
                else:
                    lease_time = 0
                job_info = JobInfo(
                    lease_time=lease_time, job_gpu_demand=job["spec"]["num_replicas"]
                )
                job_info.restarts = job.get("status", {}).get("group") or 0

                if job["spec"]["strategy"] == "zero-offload":
                    cpu_demand = 4 * job["spec"]["num_replicas"]
                    mem_demand = 8 * job["spec"]["num_replicas"]
                else:
                    cpu_demand = 1 * job["spec"]["num_replicas"]
                    mem_demand = 2 * job["spec"]["num_replicas"]

                job_info.job_demand_vector = [
                    job["spec"]["num_replicas"],
                    cpu_demand,
                    mem_demand,
                    0,
                ]
                job_info.strategy = job["spec"]["strategy"]
                if job_info.strategy not in ["ga", "zero-offload", "zero-dp", "gc"]:
                    jobModel = MODELS[job["spec"]["application"]]
                    job_info.avail_placement = jobModel.avail_placement[
                        job_info.strategy
                    ]
                else:
                    self.avail_placement = None
            else:
                job_info = JobInfo(
                    model_name=job["spec"]["application"],
                    local_bsz=job["spec"]["localBsz"],
                    rsc_metrix=self.rsc_metrix[name],
                    max_throughput_config=self.max_throughput_config[name],
                    origin_tpt=job["spec"]["origTpt"],
                    preemptible=True,
                )
                job_info.model_name = job["spec"]["application"]
                job_info.steps = job.get("status", {}).get("train", {}).get("steps", 0)
                job_info.restarts = job.get("status", {}).get("group") or 0
                job_info.is_distributed = (
                    job.get("status", {}).get("is_distributed") or False
                )
                job_info.job_demand_vector = [job["spec"]["num_replicas"]]
                job_info.job_gpu_demand = job["spec"]["num_replicas"]
                job_info.strategy = job["spec"]["strategy"]
                job_info.is_sla = job["spec"]["is_sla"]
                jobModel = MODELS[job["spec"]["application"]]

                if job["spec"]["application"] not in ["llama7", "llama30"] or (
                    job["spec"]["application"] == "llama7"
                    and job["spec"]["strategy"] == "zero-offload"
                ):
                    p = tuple([job["spec"]["num_replicas"]])
                else:
                    p = tuple([jobModel.avail_placement[job_info.strategy][0]])
                if job_info.is_sla:
                    job_info.sla_perf = (
                        job["spec"]["localBsz"]
                        / APPLICATIONS[job["spec"]["application"]].get_throughput(
                            p, job_info.strategy
                        )[2]
                    )
                else:
                    job_info.sla_perf = 0
                if job_info.strategy not in ["ga", "zero-offload", "zero-dp", "gc"]:
                    job_info.avail_placement = jobModel.avail_placement[
                        job_info.strategy
                    ]
                    job_info.expand_placement = []
                    for i in range(1, 9):
                        s = job_info.strategy[:2] + str(i)
                        if s in jobModel.avail_placement:
                            job_info.expand_placement += jobModel.avail_placement[s]
                else:
                    job_info.avail_placement = None
                    job_info.expand_placement = None
                job_info.age = (
                    datetime.now()
                    - datetime.strptime(
                        job["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    )
                ).total_seconds()

            job_infos[(namespace, name)] = job_info
        return job_infos, allocations

    def _allocate(self, jobs, nodes, prev_allocations):
        for job_key in list(jobs):
            allocations = {}
        allocations, desired_nodes = self._policy.optimize(
            jobs, nodes, prev_allocations
        )
        return allocations


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    allocator = MorphlingAllocator()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            allocator.run(),
        )
    )
    loop.close()
