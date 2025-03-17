import os
import sys
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
sched_dir = os.path.abspath(os.path.join(current_dir, "../../sched"))
if sched_dir not in sys.path:
    sys.path.append(sched_dir)

os.chdir(current_dir)

from morphling_sched.policy.morphling import MorphlingPolicy
from job_info import JobInfo, NodeInfo
from models.roberta import roberta
from morphling_sched.profile_forward import build_model
from morphling_sched.pred_throughput import ThroughputFunction
from morphling_sched.performance_model import PerformanceModel
from t5_3B_config import t5_3B


class MorphlingScheduler:
    def __init__(self, num_gpus=0):
        self.num_gpus = num_gpus
        self.num_jobs = 0

        self.jobs = []
        self.allocation = {}

        self.node_infos = self.get_node_infos(num_nodes=1, num_gpus=self.num_gpus)
        self.job_infos = {}

        # evaluate the Scheduling Policy of Morphling
        self.policy = MorphlingPolicy()

    def submit(self, job_name):
        self.jobs.append(job_name)
        self.num_jobs += 1
        # Special case handling because the RoBERTa model
        # can be directly obtained from the simulator/models.
        if job_name == "roberta":
            args = self.submit_roberta_model(job_name)
        else:
            args = self.submit_t5_3B_model(job_name)
        self.job_infos[job_name] = self.get_job_infos(job_name, *args)

    def allocate(self):
        alloc_result = {"gpu": {}, "cpu": {}, "mem": {}, "exec_plan": {}}
        alloc_result = self.policy.optimize(
            self.job_infos, self.node_infos, alloc_result
        )[0]
        for job in self.jobs:
            self.allocation[job] = (
                len(alloc_result["gpu"][job]),
                transform_exec_plan(alloc_result["exec_plan"][job]),
            )
        return self.allocation

    def simulate(self):
        job_performance = {}
        for job in self.jobs:
            job_trace = "../../simulator/traces/" + job + "/placements.csv"
            job_performance.setdefault(job, [100.0, 100.0])
            with open(job_trace, "r", encoding="utf-8-sig") as data:
                reader_config = csv.DictReader(data)
                data_rows = list(reader_config)
            for data_row in data_rows:
                if data_row["placement"] == str(self.allocation[job][0]):
                    if job_performance[job][0] > float(data_row["iter_time"]):
                        job_performance[job][0] = float(data_row["iter_time"])
                        job_performance[job][1] = data_row["exec_plan"]
        return job_performance

    def get_node_infos(self, num_nodes, num_gpus):
        return {
            idx: NodeInfo(
                {
                    "gpu": num_gpus,
                    "cpu": 100,
                    "mem": 1000,
                    "intra_bandwidth": 100,
                    "inter_bandwidth": 20,
                },
                preemptible=False,
            )
            for idx in range(num_nodes)
        }

    def submit_roberta_model(self, job_name):
        roberta_model = roberta("roberta")
        forward_time, model_flops, parameter_size = build_model(
            job_name,
            roberta_model.hidden_size,
            roberta_model.num_attention_heads,
            roberta_model.num_hidden_layers,
            1,
        )
        return roberta_model, forward_time, model_flops, parameter_size

    def submit_t5_3B_model(self, job_name):
        t5_3B_model = t5_3B()
        t5_3B_model.model_info()
        return t5_3B_model, 500, 0, 3338423808 / 1024 / 1024

    def get_job_infos(self, job_name, model, forward_time, model_flops, parameter_size):
        throughput_fn = ThroughputFunction(model.perf_params, model.env_params)
        perf_model = PerformanceModel(
            job_name,
            parameter_size,
            model.local_bsz,
            forward_time,
            model.sequence_length * model.hidden_size,
            model.num_hidden_layers,
            model.max_dp_atom_bsz,
            model.max_mp_atom_bsz,
            model.avail_placement,
            model.env_params,
            model.perf_params,
        )
        perf_model.max_gpus = 4
        perf_model.gen_perf_curve(job_name)

        job_info = JobInfo(
            model_name=job_name,
            parameter_size=parameter_size,
            forward_time=forward_time,
            local_bsz=model.local_bsz,
            res_matrix=perf_model.res_matrix,
            max_throughput_config=perf_model.max_throughput_config,
            throughput=0,
            steps=10000,
            restarts=0,
            throughput_fn=throughput_fn,
            is_distributed=False,
            origin_tpt=1,
            preemptible=True,
        )
        job_info.age = 1
        job_info.is_sla = False
        job_info.gpu_to_placements = getattr(model, "gpu_to_placements", None)
        job_info.placements_to_plan = getattr(model, "placements_to_plan", None)
        return job_info


# Just transform the execution plan to the format of the csv.
def transform_exec_plan(origin_plan):
    if "zero-offload" in origin_plan:
        exec_plan = "zero-offload"
    elif "zero-dp" in origin_plan:
        exec_plan = "zero-dp"
    elif origin_plan[0].isdigit():
        if origin_plan[0][0] == origin_plan[0][2] == "1":
            exec_plan = "tp+ga"
        elif origin_plan[0][1] == origin_plan[0][2] == "1":
            exec_plan = "pp+ga"
        elif origin_plan[0][0] == origin_plan[0][1] == "1":
            exec_plan = "dp+ga"
        else:
            exec_plan = "3D"
    else:
        exec_plan = "dp"
    if "gc" in origin_plan:
        exec_plan += "+gc"
    elif "ga" in origin_plan:
        exec_plan += "+ga"
    return exec_plan
