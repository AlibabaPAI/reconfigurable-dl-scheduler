import json
import os


def allowed_taints(taints):
    if not taints:
        return True
    if (
        len(taints) == 1
        and taints[0].key == "alibaba.com/nodegroup"
        and taints[0].value == "scheduler"
    ):
        return False


def get_namespace():
    if not os.path.exists(
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    ):  # noqa: E501
        return "default"
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        return f.read()


def get_image():
    return os.environ["MORPHLING_IMAGE"]


def get_morphling_deployment():
    return os.environ["MORPHLING_SCHED_DEPLOYMENT"]


def get_job_default_resources():
    val = os.getenv("MORPHLING_JOB_DEFAULT_RESOURCES")
    return json.loads(val) if val is not None else None


def get_job_patch_pods():
    val = os.getenv("MORPHLING_JOB_PATCH_PODS")
    return json.loads(val) if val is not None else None


def get_job_patch_containers():
    val = os.getenv("MORPHLING_JOB_PATCH_CONTAINERS")
    return json.loads(val) if val is not None else None
