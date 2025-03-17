#!/usr/bin/env python3

import argparse
import copy
import itertools
import os
import pandas
import subprocess
import time
import yaml
from kubernetes import client, config, watch


def build_images(models):
    # Build image for each model and upload to registry.
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    templates = {}
    for model in models:
        with open(os.path.join(models_dir, model, "morphlingjob.yaml")) as f:
            template = yaml.safe_load(f)
        templates[model] = template
    return templates


def cache_images(templates):
    # Cache job images on all nodes in the cluster.
    daemonset = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": "images"},
        "spec": {
            "selector": {"matchLabels": {"name": "images"}},
            "template": {
                "metadata": {"labels": {"name": "images"}},
                "spec": {
                    "containers": [],
                    "imagePullSecrets": [{"name": "mycred"}],
                },
            },
        },
    }
    for name, template in templates.items():
        daemonset["spec"]["template"]["spec"]["containers"].append(
            {
                "name": name,
                "image": template["spec"]["template"]["spec"]["containers"][0]["image"],
                "command": ["sleep", "1000000000"],
            }
        )
    apps_api = client.AppsV1Api()
    namespace = config.list_kube_config_contexts()[1]["context"].get(
        "namespace", "default"
    )
    apps_api.create_namespaced_daemon_set(namespace, daemonset)
    while True:
        # Wait for DaemonSet to be ready.
        obj = apps_api.read_namespaced_daemon_set("images", namespace)
        ready = obj.status.number_ready
        total = obj.status.desired_number_scheduled
        print("caching images on all nodes: {}/{}".format(ready, total))
        if total > 0 and ready >= total:
            break
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "policy",
        type=str,
        choices=["pollux", "morphling", "synergy", "morphling-N", "antman"],
        default="morphling",
    )
    parser.add_argument("workload", type=str, help="path to workload csv")
    args = parser.parse_args()

    workload = pandas.read_csv(args.workload)

    config.load_kube_config()

    templates = build_images(
        ["roberta", "vit", "llama7", "llama30", "bert", "gpt2", "t5"]
    )

    objs_api = client.CustomObjectsApi()
    namespace = config.list_kube_config_contexts()[1]["context"].get(
        "namespace", "default"
    )
    obj_args = ("morphling.alibaba.com", "v1", namespace, "morphlingjob")

    print("start workload")
    start = time.time()
    for row in workload.sort_values(by="submit_time").itertuples():
        while time.time() - start < row.submit_time:
            time.sleep(1)
        print("submit job {} at time {}".format(row, time.time() - start))
        job = copy.deepcopy(templates[row.application])
        job["metadata"].pop("generateName")
        job["metadata"]["name"] = row.name
        job["spec"].update(
            {
                "application": row.application,
                "steps": row.steps,
            }
        )
        job["spec"]["strategy"] = str(row.strategy)
        job["spec"]["num_replicas"] = row.num_gpus
        job["spec"]["runtime"] = int(row.runtime)
        if int(row.isSLA) == 1:
            job["spec"]["isSLA"] = True
        else:
            job["spec"]["isSLA"] = False
        objs_api.create_namespaced_custom_object(*obj_args, job)
