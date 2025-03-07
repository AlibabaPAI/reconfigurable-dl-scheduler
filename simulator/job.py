import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sched_dir = os.path.abspath(os.path.join(current_dir, "../sched"))
if sched_dir not in sys.path:
    sys.path.append(sched_dir)

from model import Model
import numpy as np
import math
from morphling_sched.policy.morphling import MorphlingPolicy

lease_time_alloc = 60
id_num = 0


def get_id():
    global id_num
    id_num = id_num + 1
    return id_num


class Job(object):
    def __init__(
        self,
        policy,
        name,
        submission_time,
        duration,
        application,
        model=None,
        gpu_demand=1,
        exec_plan="DDP",
        completion_steps=None,
        online_model_fitting=False,
        is_sla=False,
    ):
        self.policy = policy
        self.name = name
        self.submission_time = submission_time
        self.current_time = 0.0
        self.completion_time = None

        self.application = application
        self.model = model
        self.gpu_demand = gpu_demand
        self.online_model_fitting = online_model_fitting
        self.is_sla = is_sla

        self.has_profile = False
        self.id = get_id()
        self.reallocate_time = 0
        self.steps = 0
        self.progress = 0.0

        placement = ()
        while sum(placement) < gpu_demand:
            placement = (*placement, min(gpu_demand - sum(placement), 8))
        # unused execution plans for large models
        if self.model in ["gpt2", "t5"] and exec_plan not in [
            "ga",
            "gc",
            "zero-offload",
            "zero-dp",
        ]:
            placement = (int(exec_plan[1]),) * int(exec_plan[0])
        elif self.model in ["llama7", "llama30"] and exec_plan not in [
            "ga",
            "gc",
            "zero-offload",
            "zero-dp",
        ]:
            placement = (int(exec_plan[1]),) * (int(exec_plan[0]) * int(exec_plan[2]))
        self.completion_steps = (
            completion_steps
            if completion_steps
            else self.application.get_completion_steps(duration, placement, exec_plan)
        )

        self.placement = ()
        self.num_restarts = None
        self.target_batch_size = None

        if isinstance(self.policy, MorphlingPolicy):
            self.get_res_matrix(self.model)

        if self.model != None:
            self.parameter_size = self.model.parameter_size
            self.forward_time = self.model.forward_time
            self.local_bsz = self.target_batch_size = self.model.local_bsz

            # synergy
            if exec_plan not in self.model.cpu_strategy:
                self.job_cpu_demand_orig = self.gpu_demand
                self.job_mem_demand_orig = 2 * self.gpu_demand
                self.job_sspeed_demand_orig = 2 * self.gpu_demand
            else:
                self.job_cpu_demand_orig = (
                    self.model.cpu_strategy[exec_plan] * self.gpu_demand
                )
                self.job_mem_demand_orig = (
                    self.model.mem_strategy[exec_plan] * self.gpu_demand
                )
                self.job_sspeed_demand_orig = (
                    self.model.sspeed_strategy[exec_plan] * self.gpu_demand
                )
            self.job_cpu_demand = 0
            self.job_mem_demand = 0
            self.job_sspeed_demand = 0
            self.synergy_speedup = self.model.synergy_speedup
            self.synergy_speedup_orig = self.model.synergy_speedup

        self.throughput = 0
        if self.model not in ["llama7", "llama30"] or (
            self.model == "llama7" and exec_plan == "zero-offload"
        ):
            p = tuple([gpu_demand])
        else:
            p = tuple([model.avail_placement[exec_plan][0]])
        self.orig_throughput = self.model.orig_tpt
        if self.is_sla:
            self.sla_perf = (
                self.local_bsz / self.application.get_throughput(p, exec_plan)[2]
            )
        else:
            self.sla_perf = 0

        self.exec_plan = str(exec_plan)
        if exec_plan not in ["ga", "zero-offload", "zero-dp", "gc"]:
            self.avail_placement = self.model.avail_placement[self.exec_plan]
            self.expand_placement = []
            for i in range(1, 9):
                s = exec_plan[:2] + str(i)
                if s in self.model.avail_placement:
                    self.expand_placement += self.model.avail_placement[s]
        else:
            self.avail_placement = None
            self.expand_placement = None

        # pollux
        self.pollux_perf_params = None
        self.attained_service = 0
        self.pollux_profile = {}
        self.accum_steps = 0
        self.atomic_bsz = 0

        self.lease_time = 0
        self.gpus = 0
        self.is_distributed = False

    def step(self, seconds, used_gpus, used_cpus, used_mem):
        if not self.placement:
            # No resources are allocated to the job.
            self.current_time += seconds
            return
        if self.lease_time > lease_time_alloc:
            self.lease_time = 0
        self.lease_time += seconds
        delay = min(self.reallocate_time, seconds)
        self.current_time += delay
        self.reallocate_time -= delay
        seconds -= delay
        self.attained_service += delay * sum(self.placement)
        if isinstance(self.policy, MorphlingPolicy):
            placement = tuple(filter(None, self.placement))
            num_nodes, num_gpus = len(placement), sum(placement)
            overlap_sync, optimize_time, iteration_time = (
                self.application.get_throughput(placement, self.exec_plan)
            )
            self.throughput = self.local_bsz / iteration_time
            while seconds > 0 and self.completion_time is None:
                # online update performance model
                if self.online_model_fitting:
                    num_threads = len(used_cpus) / len(used_gpus)
                    _, st_params, _ = self.model.model_solver.transform_strategy(
                        self.exec_plan, len(used_gpus)
                    )
                    self.model.profile[
                        num_nodes,
                        num_gpus,
                        self.ps,
                        self.local_bsz,
                        self.model.model_solver.seq_hid,
                        num_threads,
                    ] = (
                        self.forward_time,
                        overlap_sync,
                        optimize_time,
                        iteration_time,
                        st_params.zero_type,
                        st_params.is_mp,
                        [k for k in st_params],
                    )
                    self.model.update_params()
                # TODO: cancel perfcurve to accelerate simulator
                if not self.has_profile:
                    # self.model.model_solver.perfCurve()
                    self.has_profile = True
                if self.progress + seconds < iteration_time:
                    self.progress += seconds
                    self.current_time += seconds
                    seconds = 0
                else:
                    self.steps += 1
                    delta = float(iteration_time - self.progress)
                    if math.isnan(delta):
                        print(self.model, placement, self.exec_plan)
                    #     delta=1
                    assert delta <= seconds
                    if self.steps > self.completion_steps:
                        self.completion_time = self.current_time + delta
                    self.progress = 0
                    self.current_time = self.current_time + delta
                    seconds = seconds - delta
            self.current_time = round(self.current_time + seconds, 1)

    def reallocate(self, placement, is_changed, exec_plan="DDP"):
        if placement:
            if self.placement != tuple(placement):
                self.placement = tuple(placement)
                self.update_local_bsz(self.placement)
                self.reallocate_time = 260  # TODO: needs to check
                if self.num_restarts is None:
                    self.num_restarts = 0
                else:
                    self.num_restarts += 1
                if exec_plan is not None:
                    self.exec_plan = exec_plan
                self.lease_time = 0
        else:
            if is_changed:
                self.placement = ()
                self.atomic_bsz = 0

    def get_res_matrix(self, model_name):
        self.throughput_fn = self.model.model_solver.get_throughput_fn()
        if (
            self.model.model_solver.res_matrix == {}
            or self.model.model_solver.max_throughput_config == None
        ):
            self.model.model_solver.gen_perf_curve(model_name)
        self.res_matrix = self.model.model_solver.res_matrix
        self.max_throughput_config = self.model.model_solver.max_throughput_config

    @property
    def max_profiled_replicas(self):
        return max((k[1] for k in self.pollux_profile), default=0)

    def update_local_bsz(self, placement):
        app = self.model
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if batch_size is not None:
            return
        if batch_size is None and self.pollux_perf_params is None:
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes,
                num_replicas,
                app.max_batch_size,
                (app.min_local_bsz, app.max_local_bsz),
                accumulation=True,
            )
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))

    def update_pollux_params(
        self,
        num_nodes,
        num_replicas,
        local_bsz,
        step_time,
        sync_time,
        grad_sqr,
        grad_var,
    ):
        self.grad_params = (grad_sqr, grad_var)
        if (num_nodes, num_replicas, local_bsz) in self.profile:
            return
        self.pollux_profile[num_nodes, num_replicas, local_bsz] = step_time, sync_time
        num_nodes = np.array([key[0] for key in self.pollux_profile])
        num_replicas = np.array([key[1] for key in self.pollux_profile])
        local_bsz = np.array([key[2] for key in self.pollux_profile])
        step_time = np.array([val[0] for val in self.pollux_profile.values()])
        sync_time = np.array([val[1] for val in self.pollux_profile.values()])
        compute_time = step_time - sync_time
