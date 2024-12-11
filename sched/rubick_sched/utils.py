import kubernetes_asyncio as kubernetes


async def patch_job_status(obj_api, namespace, name, patch):
    try:
        return await obj_api.patch_namespaced_custom_object_status(
            "rubick.alibaba.com", "v1", namespace, "rubickjobs", name, patch)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 404:
            return None
        raise
