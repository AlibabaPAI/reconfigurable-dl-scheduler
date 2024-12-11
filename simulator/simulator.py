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
from simulator.rubick import RubickPolicy
from applications import APPLICATIONS
from model import MODELS
from pred_throughput import fit_perf_params, ThroughputFunction
from utils.utils import JobInfo, NodeInfo
from job import Job
from synergy import SynergyPolicy

class Cluster(object):
    def __init__(self, workload, policy, num_nodes, num_gpus, num_cpus, speed_bandwidth,mem_per_node,sspeed_per_node,network,
                 interference=0.0,
                 low_util=None, high_util=None):
        assert 1 <= num_gpus <= 8
        self.workload = workload
        self.policy = policy
        # Node Spec
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.speed_bandwidth = speed_bandwidth
        self.mem_per_node=mem_per_node
        self.sspeed_per_node=sspeed_per_node
        self.network=network

        self.interference = interference
        self.low_util = low_util
        self.high_util = high_util
        self.current_time = 0
        if isinstance(policy, RubickPolicy):
            self.jobs = [Job(row.name, APPLICATIONS[row.application], row.submit_time, row.runtime,policy,MODELS[row.application],row.application, row.num_gpus,row.strategy)
                         for row in workload.itertuples()]
            self.execution_plan = {"gpu": {}, "cpu": {}, "bandwidth": {},"strategy":{}}
        elif isinstance(policy,SynergyPolicy):
            self.jobs=[Job(row.name, APPLICATIONS[row.application], row.submit_time,row.runtime,policy,MODELS[row.application],job_gpu_demand=row.num_gpus,strategy=row.strategy)
                         for row in workload.itertuples()]
            self.execution_plan = {"gpu": {}, "cpu": {}, "mem": {},"sspeed":{},"network":{}}
            update_speed = False
            for job in self.jobs:
                if job.job_cpu_demand_orig>self.num_cpus:
                    job.job_cpu_demand_orig = self.num_cpus
                    update_speed = True
                    if job.job_gpu_demand > self.num_gpus:
                        job.job_cpu_demand_orig = job.job_gpu_demand *(self.num_cpus/self.num_gpus)
                        job.synergy_speedup = 1
                        update_speed = False
                
                if job.job_mem_demand_orig > self.mem_per_node:
                    job.job_mem_demand_orig = self.mem_per_node
                    update_speed = True
                    if job.job_gpu_demand > self.num_gpus:
                        job.job_mem_demand_orig = job.job_gpu_demand *(self.mem_per_node/self.num_gpus)
                        job.synergy_speedup = 1
                        update_speed = False


                if job.job_sspeed_demand_orig > self.sspeed_per_node:
                    job.job_sspeed_demand_orig = self.sspeed_per_node
                    update_speed = True
                    if job.job_gpu_demand > self.num_gpus:
                        job.job_sspeed_demand_orig = job.job_gpu_demand *(self.sspeed_per_node/self.num_gpus)
                        job.synergy_speedup = 1
                        update_speed = False


                job.job_cpu_demand = job.job_cpu_demand_orig
                job.job_mem_demand = job.job_mem_demand_orig
                job.job_sspeed_demand = job.job_sspeed_demand_orig
                job.synergy_speedup_orig = job.synergy_speedup


                    
        self.logs = []
        self.utility = []
 
    def step(self, scheduling_interval=60):
        for job in self.jobs:
            used_gpus = self.execution_plan["gpu"].get(job.name, [])
            used_cpus = self.execution_plan["cpu"].get(job.name, []) if "cpu" in self.execution_plan else used_gpus
            job.step(scheduling_interval, used_gpus, used_cpus)
            if isinstance(self.policy, RubickPolicy):
                job.get_rsc_metrix()
        self.current_time += scheduling_interval
        assert all(job.current_time == self.current_time for job in self.jobs)
        job_infos = self.get_job_infos()
        if job_infos:
            # Optimize allocations.
            node_infos = self.get_node_infos()
            for rsc in ["gpu", "cpu", "bandwidth","strategy"]:
                if rsc in self.execution_plan:
                    self.execution_plan[rsc] = {
                        k: v for k, v in self.execution_plan[rsc].items() if k in job_infos}
            # Scheduling core
            results = self.policy.optimize(job_infos, node_infos,
                                           self.execution_plan, node_infos[0])
            execution_plan,desired_nodes = results
            used_gpus = collections.Counter(
                sum(execution_plan["gpu"].values(), []))
            assert all(val <= node_infos[key].resources["gpu"]
                       for key, val in used_gpus.items())
            for job in self.jobs:
                alloc = execution_plan["gpu"].get(job.name, [])
                
                if len(alloc)==0:
                    continue
                placement = []
                for i in range(len(alloc)):
                    if i == 0 or alloc[i] != alloc[i - 1]:
                        placement.append(1)
                    else:
                        placement[-1] += 1
                if isinstance(self.policy, RubickPolicy):
                    job.reallocate(placement, execution_plan["gpu"].get(job.name) != self.execution_plan["gpu"].get(job.name),len(execution_plan["cpu"].get(job.name)),execution_plan["strategy"].get(job.name))
                else:
                    job.reallocate(placement, execution_plan["gpu"].get(job.name) != self.execution_plan["gpu"].get(job.name),len(execution_plan["cpu"].get(job.name)))
                if len(placement)>1:
                    job.is_distributed=True
                if "strategy" in execution_plan:
                    job.strategy = execution_plan["strategy"].get(job.name)
            self.execution_plan = execution_plan
        self.logs.append({
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "num_restarts": job.num_restarts,
                    "allocation": self.execution_plan.get(job.name, []),
                    "placement": job.placement,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        })

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, RubickPolicy):
                    job_infos[job.name] = self.get_mysys_job_info(job)
                elif isinstance(self.policy, SynergyPolicy):
                    job_infos[job.name] = self.get_synergy_job_info(job) 
        return job_infos
        

    def get_mysys_job_info(self, job):
        job_info = JobInfo(model_name=job.model_name,ps=job.ps,forward_time=job.forward_time,local_bsz=job.local_bsz,
            rsc_metrix=job.rsc_metrix, 
            max_throughput_config=job.max_throughput_config, throughput=job.throughput,
            creation_timestamp=job.submission_time,steps=job.steps, restarts=job.num_restarts,throughput_fn=job.throughput_fn,is_distributed=job.is_distributed,origin_tpt=job.orig_throughput,
            preemptible=True,
        )
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info    

    def get_synergy_job_info(self, job):
        job_info = JobInfo(
            lease_time=job.lease_time,job_gpu_demand=job.job_gpu_demand
        )
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        job_info.job_demand_vector=[job.job_gpu_demand , job.job_cpu_demand , job.job_mem_demand,job.job_sspeed_demand,0]
        return job_info

    def get_node_infos(self, num_nodes=None):
        return {
            idx: NodeInfo({"gpu": self.num_gpus, "cpu": self.num_cpus,
                          "bandwidth": self.speed_bandwidth,
                          "mem":self.mem_per_node,
                          "sspeed":self.sspeed_per_node,
                          "network":self.network},
                          preemptible=False)
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
        return {
            val["name"]: val["completion_time"] - val["submission_time"]
            for val in self.logs[-1]["submitted_jobs"]
            if val["completion_time"] is not None
        }


def simulate(args):
    workload = pandas.read_csv(args.workload)
    if args.policy == "mysys":
        policy = RubickPolicy() 
    elif args.policy=="synergy":
        policy=SynergyPolicy()
    simulator = Cluster(workload, policy, args.num_nodes, num_gpus=args.num_gpus, num_cpus=args.num_cpus, speed_bandwidth=args.speed_bandwidth,
                        mem_per_node=args.mem_per_node,sspeed_per_node=args.sspeed_per_node,network=args.network_node,
                        interference=args.interference,
                        low_util=args.low_util, high_util=args.high_util)
    while not simulator.all_complete():
        simulator.step(args.interval)
        print("---------------- SIMULATOR TIME: {} ----------------"
              .format(simulator.current_time))
        print("Active jobs:")
        for val in simulator.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= simulator.current_time and val["completion_time"] is None:
                print("    {}:\t[restarts {}]\t[placement {}]".format(
                      val["name"],  val["num_restarts"],  val["placement"]))
        used_gpus = sum(map(len, simulator.execution_plan["gpu"].values()))
        print("GPU utilization: {}".format(used_gpus))
        print("Completed jobs:")
        jct_dict = simulator.get_jcts()
        print(jct_dict)
        print("Average JCT:", sum(jct_dict.values()) /
              len(jct_dict) if jct_dict else 0)
    if args.output:
        simulator.output_logs(args.output)
    return simulator.logs, simulator.get_jcts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--policy", type=str, default="mysys",
                        choices=["mysys", "synergy", "sia", "antman","mysysSLA"])
    parser.add_argument("--num-nodes", type=int, default=8,
                        help="number of nodes in the cluster")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--interference", type=float, default=0.0,
                        help="job slowdown due to interference")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="number of GPUs per node")
    parser.add_argument("--num-cpus", type=int, default=64,
                        help="number of CPUs per node")
    parser.add_argument("--speed-bandwidth", type=int, default=10,
                        help="speed of bandwidth per node")
    parser.add_argument("--mem-per-node", type=int, default=256,
                        help="memory per node")
    parser.add_argument("--sspeed-per-node", type=int, default=64,
                        help="sspeed per node")
    parser.add_argument("--network-node", type=int, default=40,
                        help="network of  node")
    parser.add_argument("--low-util", type=float,
                        help="low utility threshold")
    parser.add_argument("--high-util", type=float,
                        help="high utility threshold")
    parser.add_argument("--output", type=str,
                        help="path to output logs")
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
