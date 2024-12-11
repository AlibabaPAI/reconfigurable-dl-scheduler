from rubick_sched.pred_throughput import ThroughputFunction, StrategyParams, PerfParams
from collections import defaultdict
import json
from itertools import product
import os
import math
import copy


class ModelSolver(object):
    def __init__(self, name, ps, local_bsz, forward_time, seq_hid, max_ddp_atom_bsz, max_mp_atom_bsz):
        self.name = name
        self.ps = ps
        self.local_bsz = local_bsz
        self.forward_time = forward_time
        self.seq_hid = seq_hid
        self.max_ddp_atom_bsz = max_ddp_atom_bsz
        self.max_mp_atom_bsz = max_mp_atom_bsz

        self.min_gpus, self.max_gpus = 1, 64
        self.min_threads, self.max_threads = 1, 64
        self.min_bandwidth, self.max_bandwidth = 0, 10
        self.rsc_interval = 1
        self.bandwidth_interval = 1
        self.perf_params = PerfParams(
            k_bct_fwd=2, k_c=0.8, k_os=0.00010635832685610605, k_os_offload=1, b_overhead=0.1, k_off=1, k_swap=1)
        self.rsc_metrix = defaultdict(lambda: defaultdict(dict))
        self.set_default_rsc_remix()
        self.gpu_slope = defaultdict(lambda: defaultdict(dict))
        self.cpu_slope = defaultdict(lambda: defaultdict(dict))
        self.bandwidth_slope = defaultdict(lambda: defaultdict(dict))
        self.max_throughput_config = None
        self.env_params = json.load(
            open('./test_env/env_params.json', 'r', encoding="utf-8"))

    def get_throughput_fn(self):
        return ThroughputFunction(self.perf_params, self.env_params)

    # If throughput under prediction is -1, resource is failed to be allocated.
    def get_model_throughput_with_fixed_rsc(self, num_gpus, n_thread, bandwidth):
        throughput_fn = self.get_throughput_fn()
        strategy = [["DDP", "MP"], ["ga", "gc", ""],
                    ["zero-offload", "zero-dp", ""]]
        st_comb = list(product(strategy[0], strategy[1], strategy[2]))
        max_throughput = -1
        max_strategy = []
        for st in st_comb:
            atom_bsz, st_params, can_run = self.transform_strategy(
                st, num_gpus)

            if can_run      :
                if self.name in ["vit", "roberta"]:
                    f_t = self.forward_time
                elif self.name == "swin":
                    if st[2] == "zero-dp" or st[2] == "zero-offload":
                        f_t = self.forward_time*5
                    else:
                        f_t = self.forward_time
                else:
                    f_t = self.forward_time/num_gpus
                throughput = throughput_fn.throughput(
                    self.ps, f_t, num_gpus, self.local_bsz, n_thread, bandwidth, self.seq_hid, st_params)
                if max_throughput < throughput:
                    max_throughput = throughput
                    max_strategy = st
        self.rsc_metrix[num_gpus][n_thread][bandwidth] = [max_throughput, max_strategy, {
            "nvidia.com/gpu": num_gpus, "cpu": n_thread, "bandwidth": bandwidth}]
        if num_gpus > 1 and self.rsc_metrix[num_gpus-1][n_thread][bandwidth][0]*(1+1e-2) >= self.rsc_metrix[num_gpus][n_thread][bandwidth][0]:
            self.rsc_metrix[num_gpus][n_thread][bandwidth] = copy.deepcopy(
                self.rsc_metrix[num_gpus-1][n_thread][bandwidth])
        if n_thread > 1 and self.rsc_metrix[num_gpus][n_thread-1][bandwidth][0]*(1+1e-2) >= self.rsc_metrix[num_gpus][n_thread][bandwidth][0]:
            self.rsc_metrix[num_gpus][n_thread][bandwidth] = copy.deepcopy(
                self.rsc_metrix[num_gpus][n_thread-1][bandwidth])
        if bandwidth > 2 and self.rsc_metrix[num_gpus][n_thread][bandwidth-1][0]*(1+1e-2) >= self.rsc_metrix[num_gpus][n_thread][bandwidth][0]:
            self.rsc_metrix[num_gpus][n_thread][bandwidth] = copy.deepcopy(
                self.rsc_metrix[num_gpus][n_thread][bandwidth-1])
        return max_throughput, max_strategy

    # Draw perfCurve before starting allocation
    def perfCurve(self):
        # TODO: cut the edge
        # Structure of throughput configuration: n_gpus,n_cpus,n_bandwidth,throughput,strategy
        max_throughput_config = [0, 0, 0, 0, ["", "", ""]]
        for n_gpu in range(self.min_gpus, self.max_gpus+1, self.rsc_interval):  # 1~8
            min_threads = 1
            if n_gpu <= 8:
                max_threads = 64//n_gpu
            else:
                max_threads = 8
            for n_thread in range(min_threads, max_threads+1, self.rsc_interval):
                if n_gpu == 1:
                    min_bandwidth = max_bandwidth = 0
                elif n_gpu > 8:
                    min_bandwidth = 1
                    max_bandwidth = self.max_bandwidth
                else:
                    min_bandwidth = 0
                    max_bandwidth = self.max_bandwidth
                # Bandwidth=0 means using nvlink
                for n_banwidth in range(min_bandwidth, max_bandwidth+1, self.bandwidth_interval):
                    throughput, strategy = self.get_model_throughput_with_fixed_rsc(
                        n_gpu, n_thread, n_banwidth)
                    if max_throughput_config[3]*(1+1e-2) < throughput:
                        max_throughput_config = [
                            n_gpu, n_thread, n_banwidth, throughput, strategy]
        self.max_throughput_config = max_throughput_config

    # Strategy choice:
    # [[ DDP,           MP        ],
    #  [ ga,            gc,     ""],
    #  [ zero-offload, zero-dp, ""]]
    def transform_strategy(self, st_comb, num_gpus):
        can_run = True
        st_params = None
        if st_comb[0] == "DDP":
            mem_save_type = "DDP"
            atom_bsz = self.local_bsz/num_gpus
            if atom_bsz != int(atom_bsz):
                return 0, None, False
            if st_comb[2] == "zero-offload":
                mem_save_type = "zero-offload"
                zero_type = 1
            elif st_comb[2] == "zero-dp":
                mem_save_type = "zero-dp"
                zero_type = 2
            else:
                zero_type = 0
            if st_comb[1] == "gc":
                if atom_bsz > self.max_ddp_atom_bsz[mem_save_type][st_comb[1]]:
                    return 0, None, False
                st_params = StrategyParams(2, 1, 0, 1, 0, 0)
            else:
                if atom_bsz > self.max_ddp_atom_bsz[mem_save_type]["other"]:
                    return 0, None, False
                st_params = StrategyParams(
                    1, 1, 1, 0, zero_type)
        elif st_comb[0] == "MP":
            # These workloads do not support MP
            if self.name == "vit" or self.name == "swin" or self.name == "roberta" or num_gpus not in [1, 2, 4, 8, 16, 32]:
                return 0, None, False
            atom_bsz = self.local_bsz
            if st_comb[1] == "gc":
                if atom_bsz > self.max_mp_atom_bsz["gc"][num_gpus-1]:
                    return 0, None, False
                alpha_f = 2
                alpha_b = alpha_c = 1
            else:
                if atom_bsz > self.max_mp_atom_bsz["other"][num_gpus-1]:
                    return 0, None, False
                alpha_f = alpha_b = alpha_c = 1
            if st_comb[2] == "zero-offload" or st_comb[2] == "zero-dp":  # Not consider mp with zero
                return 0, None, False
            else:
                st_params = StrategyParams(alpha_f, alpha_b, 0, alpha_c, 1, 0)
        return atom_bsz, st_params, can_run

    def set_default_rsc_remix(self):
        for n_gpu in range(-10, self.max_gpus+10):
            for n_cpu in range(-10, self.max_threads+100):
                for n_bw in range(-10, self.max_bandwidth+100):
                    self.rsc_metrix[n_gpu][n_cpu][n_bw] = [-1, None, None]
