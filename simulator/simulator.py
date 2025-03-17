import argparse
import collections
import copy
import glob
import json
import math
import multiprocessing
import os
import numpy as np
import pandas
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sched_dir = os.path.abspath(os.path.join(current_dir, "../sched"))
if sched_dir not in sys.path:
    sys.path.append(sched_dir)

from applications import APPLICATIONS
from model import MODELS
from morphling_sched.pred_throughput import fit_perf_params, ThroughputFunction
from job_info import JobInfo, NodeInfo
from job import Job

from morphling_sched.policy.morphling import MorphlingPolicy

restart_time = {}


class Cluster(object):
    def __init__(
        self,
        workload,
        policy,
        num_nodes,
        num_gpus,
        num_cpus,
        mem_per_node,
        intra_bandwidth,
        inter_bandwidth,
        online_model_fitting,
    ):
        assert 1 <= num_gpus <= 8
        self.workload = workload
        self.policy = policy

        # Node Specification
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.mem_per_node = mem_per_node
        self.intra_bandwidth = intra_bandwidth
        self.inter_bandwidth = inter_bandwidth

        self.current_time = 0
        if isinstance(policy, MorphlingPolicy):
            self.jobs = [
                Job(
                    policy,
                    row.name,
                    row.submission_time,
                    row.duration,
                    APPLICATIONS[row.application],
                    MODELS[row.application],
                    row.num_gpus,
                    row.exec_plan,
                    row.steps,
                    online_model_fitting,
                    is_sla=row.is_sla,
                )
                for row in workload.itertuples()
            ]
            self.alloc_result = {"gpu": {}, "cpu": {}, "mem": {}, "exec_plan": {}}
        self.logs = []

    def step(self, scheduling_interval=60):
        for job in self.jobs:
            used_gpus = self.alloc_result["gpu"].get(job.name, [])
            used_cpus = (
                self.alloc_result["cpu"].get(job.name, [])
                if "cpu" in self.alloc_result
                else used_gpus
            )
            used_mem = (
                self.alloc_result["mem"].get(job.name, [])
                if "mem" in self.alloc_result
                else 0
            )
            job.step(scheduling_interval, used_gpus, used_cpus, used_mem)
        self.current_time += scheduling_interval
        for job in self.jobs:
            if job.current_time != self.current_time:
                print(job.name, job.current_time, self.current_time)
        assert all(job.current_time == self.current_time for job in self.jobs)
        job_infos = self.get_job_infos()
        if job_infos:
            # Optimize allocations.
            node_infos = self.get_node_infos()
            for res in ["gpu", "cpu", "mem", "bandwidth", "exec_plan"]:
                if res in self.alloc_result:
                    self.alloc_result[res] = {
                        k: v
                        for k, v in self.alloc_result[res].items()
                        if k in job_infos
                    }
            # scheduling core
            results = self.policy.optimize(job_infos, node_infos, self.alloc_result)
            alloc_result, desired_nodes = results
            self.alloc_result = alloc_result
            used_gpus = collections.Counter(sum(alloc_result["gpu"].values(), []))
            assert all(
                val <= node_infos[key].resources["gpu"]
                for key, val in used_gpus.items()
            )
            for job in self.jobs:
                alloc = alloc_result["gpu"].get(job.name, [])
                placement = []
                for i in range(len(alloc)):
                    if i == 0 or alloc[i] != alloc[i - 1]:
                        placement.append(1)
                    else:
                        placement[-1] += 1
                if isinstance(self.policy, MorphlingPolicy):
                    job.reallocate(
                        placement,
                        alloc_result["gpu"].get(job.name)
                        != self.alloc_result["gpu"].get(job.name),
                        exec_plan=alloc_result["exec_plan"].get(job.name),
                    )
                else:
                    job.reallocate(
                        placement,
                        alloc_result["gpu"].get(job.name)
                        != self.alloc_result["gpu"].get(job.name),
                        strategy=job.strategy,
                    )
                if len(placement) > 1:
                    job.is_distributed = True
        self.logs.append(
            {
                "timestamp": self.current_time,
                "num_nodes": self.num_nodes,
                "submitted_jobs": [
                    {
                        "name": job.name,
                        "num_restarts": job.num_restarts,
                        "allocation": self.alloc_result.get(job.name, []),
                        "placement": job.placement,
                        "submission_time": job.submission_time,
                        "completion_time": job.completion_time,
                    }
                    for job in self.jobs
                    if job.submission_time <= self.current_time
                ],
            }
        )

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, MorphlingPolicy):
                    job_infos[job.name] = self.get_morphling_job_info(job)
        return job_infos

    def get_morphling_job_info(self, job):
        job_info = JobInfo(
            model_name=job.model.model_name,
            parameter_size=job.parameter_size,
            forward_time=job.forward_time,
            local_bsz=job.local_bsz,
            res_matrix=job.res_matrix,
            max_throughput_config=job.max_throughput_config,
            throughput=job.throughput,
            steps=job.steps,
            restarts=job.num_restarts or 0,
            throughput_fn=job.throughput_fn,
            is_distributed=job.is_distributed,
            origin_tpt=job.orig_throughput,
            preemptible=True,
        )
        job_info.age = self.current_time - job.submission_time
        job_info.gpu_to_placements = job.model.gpu_to_placements
        job_info.placements_to_plan = job.model.placements_to_plan
        job_info.gpu_demand = job.gpu_demand
        job_info.is_sla = job.is_sla
        job_info.sla_perf = job.sla_perf
        return job_info

    def get_synergy_job_info(self, job):
        job_info = JobInfo(
            gpu_demand=job.gpu_demand,
            restarts=job.num_restarts or 0,
        )
        job_info.lease_time = job.lease_time
        job_info.age = self.current_time - job.submission_time
        job_info.job_demand_vector = [
            job.gpu_demand,
            job.job_cpu_demand,
            job.job_mem_demand,
            job.job_sspeed_demand,
            0,
        ]
        job_info.strategy = job.strategy
        job_info.avail_placement = job.avail_placement
        return job_info

    def get_antman_job_info(self, job):
        job_info = JobInfo(
            gpu_demand=job.gpu_demand,
            restarts=job.num_restarts or 0,
        )
        job_info.age = self.current_time - job.submission_time
        job_info.strategy = str(job.strategy)
        job_info.avail_placement = job.avail_placement
        job_info.is_sla = job.is_sla
        job_info.model_name = job.model_name
        return job_info

    def get_morphing_res_job_info(self, job):
        job_info = JobInfo(
            model_name=job.model_name,
            ps=job.ps,
            forward_time=job.forward_time,
            local_bsz=job.local_bsz,
            res_matrix=job.res_matrix,
            gpu_demand=job.gpu_demand,
            max_throughput_config=job.max_throughput_config,
            throughput=job.throughput,
            steps=job.steps,
            restarts=job.num_restarts or 0,
            throughput_fn=job.throughput_fn,
            is_distributed=job.is_distributed,
            origin_tpt=job.orig_throughput,
            preemptible=True,
        )
        job_info.age = self.current_time - job.submission_time
        job_info.model_name = job.model_name
        job_info.strategy = job.strategy
        job_info.avail_placement = job.avail_placement
        job_info.expand_placement = job.expand_placement
        return job_info

    def get_morphling_none_job_info(self, job):
        job_info = JobInfo(
            model_name=job.model_name,
            ps=job.ps,
            forward_time=job.forward_time,
            local_bsz=job.local_bsz,
            res_matrix=job.res_matrix,
            gpu_demand=job.gpu_demand,
            max_throughput_config=job.max_throughput_config,
            throughput=job.throughput,
            steps=job.steps,
            restarts=job.num_restarts or 0,
            throughput_fn=job.throughput_fn,
            is_distributed=job.is_distributed,
            origin_tpt=job.orig_throughput,
            preemptible=True,
        )
        job_info.age = self.current_time - job.submission_time
        job_info.job_demand_vector = [job.gpu_demand]
        job_info.model_name = job.model_name
        job_info.strategy = job.strategy
        job_info.avail_placement = job.avail_placement
        return job_info

    def get_pollux_job_info(self, job):
        job_info = JobInfo(
            resources={"gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=min(
                max(2 * job.max_profiled_replicas, 1),
                64,  # simulator can't handle more.
                job.application.max_batch_size // job.application.min_local_bsz,
            ),
            restarts=job.num_restarts or 0,
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.creation_timestamp = (job.submission_time,)
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_node_infos(self, num_nodes=None):
        return {
            idx: NodeInfo(
                {
                    "gpu": self.num_gpus,
                    "cpu": self.num_cpus,
                    "mem": self.mem_per_node,
                    "intra_bandwidth": self.intra_bandwidth,
                    "inter_bandwidth": self.inter_bandwidth,
                },
                preemptible=False,
            )
            for idx in range(num_nodes or self.num_nodes)
        }

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_jcts(self):
        for val in self.logs[-1]["submitted_jobs"]:
            if val["completion_time"] is not None:
                restart_time[val["name"]] = val["num_restarts"]

        return {
            val["name"]: val["completion_time"] - val["submission_time"]
            for val in self.logs[-1]["submitted_jobs"]
            if val["completion_time"] is not None
        }


def simulate(args):
    workload = pandas.read_csv(args.workload)

    policy_mapping = {
        "morphling": MorphlingPolicy,
    }
    if args.policy in policy_mapping:
        policy = policy_mapping[args.policy]()
    else:
        raise ValueError(f"Unknown Scheduling Policy: {args.policy}")

    simulator = Cluster(
        workload,
        policy,
        args.num_nodes,
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
        mem_per_node=args.mem_per_node,
        inter_bandwidth=args.inter_bandwidth,
        intra_bandwidth=args.intra_bandwidth,
        online_model_fitting=args.online_model_fitting,
    )
    while not simulator.all_complete():
        simulator.step(args.interval)
        print(
            "---------------- SIMULATOR TIME: {} ----------------".format(
                simulator.current_time
            )
        )
        print("Active jobs:")
        for val in simulator.logs[-1]["submitted_jobs"]:
            if (
                val["submission_time"] <= simulator.current_time
                and val["completion_time"] is None
            ):
                print(
                    "    {}:\t[restarts {}]\t[placement {}]".format(
                        val["name"], val["num_restarts"], val["placement"]
                    )
                )
        used_gpus = sum(map(len, simulator.alloc_result["gpu"].values()))
        print("GPU utilization: {}".format(used_gpus))
        print("Completed jobs:")
        jct_dict = simulator.get_jcts()
        print(jct_dict)
        print(
            "Average JCT (s):",
            sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0,
        )
        print("Makespan (s):", simulator.current_time)
    if args.output:
        simulator.output_logs(args.output)
    return simulator.logs, simulator.get_jcts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument(
        "--policy",
        type=str,
        default="morphling",
        choices=["morphling", "synergy", "sia", "antman"],
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="scheduling interval in seconds"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=8, help="number of nodes in the cluster"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8, help="number of GPUs per node"
    )
    parser.add_argument(
        "--num-cpus", type=int, default=96, help="number of CPUs per node"
    )
    parser.add_argument(
        "--mem-per-node", type=int, default=1600, help="memory (GB) per node"
    )
    parser.add_argument(
        "--intra-bandwidth",
        type=int,
        default=400,
        help="bandwidth (GB/s) within a node",
    )
    parser.add_argument(
        "--inter-bandwidth", type=int, default=100, help="bandwidth (GB/s) across nodes"
    )
    parser.add_argument(
        "--online-model-fitting",
        type=int,
        default=0,
        help="enable the online fitting module of performance model",
    )
    parser.add_argument("--output", type=str, help="path to output logs")
    args = parser.parse_args()

    if os.path.isdir(args.workload):
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
        with multiprocessing.Pool(processes=8) as pool:
            ret_list = pool.map(simulate, args_list)
        summary = {"jcts": {}, "avgs": {}}
        for args_item, (_, jct_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = jct_dict
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        simulate(args)
