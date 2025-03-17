def owner_reference_template(
    namespace, name, uid, kind="MorphlingContext", api="morphling.alibaba.com/v1"
):
    return [
        {
            "apiVersion": api,
            "controller": True,
            "blockOwnerDeletion": True,
            "kind": kind,
            "name": name,
            "uid": uid,
        }
    ]
