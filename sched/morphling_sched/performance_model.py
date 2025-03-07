from morphling_sched.pred_throughput import (
    ThroughputFunction,
    StrategyParams,
    PerfParams,
    EnvParams,
)
from collections import defaultdict
from itertools import product
import copy


class PerformanceModel(object):
    def __init__(
        self,
        name,
        ps,
        local_bsz,
        forward_time,
        seq_hid,
        layer_num,
        max_dp_atom_bsz,
        max_mp_atom_bsz,
        available_placement,
        env_params,
        perf_params,
    ):
        self.name = name
        self.ps = ps
        self.local_bsz = local_bsz
        self.forward_time = forward_time
        self.seq_hid = seq_hid
        self.layer_num = layer_num
        self.max_dp_atom_bsz = max_dp_atom_bsz
        self.max_mp_atom_bsz = max_mp_atom_bsz

        self.min_gpus, self.max_gpus = 1, 64
        self.min_threads, self.max_threads = 1, 64
        self.min_bandwidth, self.max_bandwidth = 0, 10
        self.res_interval = 1
        self.bandwidth_interval = 1
        self.available_placement = available_placement
        self.env_params = env_params
        self.perf_params = perf_params

        self.res_matrix = defaultdict(lambda: defaultdict(dict))
        self.set_default_res_matrix()
        self.res_matrix_3D = {}
        self.gpu_slope = defaultdict(lambda: defaultdict(dict))
        self.cpu_slope = defaultdict(lambda: defaultdict(dict))
        self.bandwidth_slope = defaultdict(lambda: defaultdict(dict))
        self.max_throughput_config = None

    def get_throughput_fn(self):
        return ThroughputFunction(self.perf_params, self.env_params)

    # value=-1: resource failed to allocate
    def get_plan_with_fixed_res(self, num_gpus, n_thread, bandwidth, placement=None):
        throughput_fn = self.get_throughput_fn()
        strategy = [["DP", "TP/PP"], ["ga", "gc"], ["zero-offload", "zero-dp", ""]]
        st_comb = list(product(strategy[0], strategy[1], strategy[2]))
        max_throughput = -1
        max_strategy = []
        for st in st_comb:
            atom_bsz, params, can_run = self.transform_strategy(st, num_gpus, placement)
            if can_run and st[0] == "DP":
                f_t = self.forward_time
                if self.name in ["bert"]:
                    if num_gpus == 2:
                        f_t = self.forward_time / 1.8
                    elif num_gpus == 4:
                        f_t = self.forward_time / 3.2
                    elif num_gpus == 8:
                        f_t = self.forward_time / 5.8
                else:
                    f_t = self.forward_time / num_gpus
                throughput = throughput_fn.throughput(
                    self.ps,
                    f_t,
                    num_gpus,
                    self.local_bsz,
                    n_thread,
                    bandwidth,
                    self.seq_hid,
                    params,
                )
                if max_throughput < throughput:
                    max_throughput = throughput
                    max_strategy = st
            elif can_run and st[0] == "TP/PP":
                if placement is not None:
                    self.res_matrix_3D.setdefault(placement, [0, 0])
                for strategy in params:
                    fct = (
                        self.forward_time
                        * strategy[2]
                        / strategy[0]
                        / pow(1.15, strategy[1])
                    )
                    throughput = throughput_fn.throughput_3D(
                        self.ps,
                        fct,
                        strategy[0],
                        strategy[1],
                        len(placement),
                        int(placement[0]),
                        atom_bsz,
                        self.seq_hid,
                        self.layer_num,
                    )
                    if throughput > self.res_matrix_3D[placement][0]:
                        self.res_matrix_3D[placement] = [throughput, strategy]
        self.res_matrix[num_gpus][n_thread][bandwidth] = [
            max_throughput,
            max_strategy,
            {"gpu": num_gpus, "cpu": n_thread, "bandwidth": bandwidth},
        ]
        if (
            num_gpus > 1
            and self.res_matrix[num_gpus - 1][n_thread][bandwidth][0] * (1 + 1e-2)
            >= self.res_matrix[num_gpus][n_thread][bandwidth][0]
        ):
            self.res_matrix[num_gpus][n_thread][bandwidth] = copy.deepcopy(
                self.res_matrix[num_gpus - 1][n_thread][bandwidth]
            )
        if (
            n_thread > 1
            and self.res_matrix[num_gpus][n_thread - 1][bandwidth][0] * (1 + 1e-2)
            >= self.res_matrix[num_gpus][n_thread][bandwidth][0]
        ):
            self.res_matrix[num_gpus][n_thread][bandwidth] = copy.deepcopy(
                self.res_matrix[num_gpus][n_thread - 1][bandwidth]
            )
        if (
            bandwidth > 2
            and self.res_matrix[num_gpus][n_thread][bandwidth - 1][0] * (1 + 1e-2)
            >= self.res_matrix[num_gpus][n_thread][bandwidth][0]
        ):
            self.res_matrix[num_gpus][n_thread][bandwidth] = copy.deepcopy(
                self.res_matrix[num_gpus][n_thread][bandwidth - 1]
            )
        return max_throughput, max_strategy

    # draw perf_curve before starting allocation
    def gen_perf_curve(self, model_name):
        # TODO: cut the edge
        # the oder of the value of throughput config:  n_gpus,n_cpus,n_bandwidth,throughput(gbsz/iter_time),strategy
        if model_name == "llama7":
            max_throughput_config = [64, 1, 1, 8 / 0.67, ("248", "248", 88888888)]
        elif model_name == "llama30":
            max_throughput_config = [64, 1, 1, 4 / 2.017, ("444", "444", 88888888)]
        else:
            max_throughput_config = [
                0,
                0,
                0,
                0,
                ["", "", ""],
            ]
            for n_gpu in range(
                self.min_gpus, self.max_gpus + 1, self.res_interval
            ):  # 1~64
                min_threads = 1
                if n_gpu <= 8:
                    max_threads = 64 // n_gpu
                else:
                    max_threads = 8
                for n_thread in range(min_threads, max_threads + 1, self.res_interval):
                    if n_gpu == 1:
                        min_bandwidth = max_bandwidth = 0
                    elif n_gpu > 8:
                        min_bandwidth = 1
                        max_bandwidth = self.max_bandwidth
                    else:
                        min_bandwidth = 0
                        max_bandwidth = self.max_bandwidth
                    for n_banwidth in range(
                        min_bandwidth, max_bandwidth + 1, self.bandwidth_interval
                    ):  # bandwidth=0 mean using nvlink
                        throughput, strategy = self.get_plan_with_fixed_res(
                            n_gpu, n_thread, n_banwidth
                        )
                        if max_throughput_config[3] * (1 + 1e-2) < throughput:
                            max_throughput_config = [
                                n_gpu,
                                n_thread,
                                n_banwidth,
                                throughput,
                                strategy,
                            ]
        self.max_throughput_config = max_throughput_config

    def transform_strategy(self, st_comb, num_gpus, placement=None):
        can_run = True
        st_params = None
        if st_comb[0] == "DP":
            mem_save_type = "DP"
            atom_bsz = self.local_bsz / num_gpus
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
                if atom_bsz > self.max_dp_atom_bsz[mem_save_type][st_comb[1]]:
                    return 0, None, False
                st_params = StrategyParams(2, 1, 1, 0, 0, zero_type)
            else:
                if atom_bsz > self.max_dp_atom_bsz[mem_save_type]["other"]:
                    return 0, None, False
                st_params = StrategyParams(1, 1, 1, 0, 0, zero_type)
        elif st_comb[0] == "TP/PP":
            # Some workloads do not need to use TP/PP.
            if (
                self.name == "vit"
                or self.name == "roberta"
                or self.name == "bert"
                # or num_gpus not in [1, 2, 4, 8, 16, 32]
            ):
                return 0, None, False
            #  Do not consider PP/TP with ZeRO-series.
            if st_comb[2] == "zero-offload" or st_comb[2] == "zero-dp":
                return 0, None, False

            # Need placement to decide 3D parallelism.
            if placement is None:
                return 0, None, False

            placement = int(placement)
            available_strategy = []
            for strategy in self.available_placement:
                if placement in self.available_placement[strategy]:
                    available_strategy.append([int(char) for char in strategy])
            atom_bsz = self.local_bsz
            return atom_bsz, available_strategy, can_run
        return atom_bsz, st_params, can_run

    def set_default_res_matrix(self):
        for n_gpu in range(-20, self.max_gpus + 10):
            for n_cpu in range(-20, self.max_threads + 100):
                for n_bw in range(-20, self.max_bandwidth + 100):
                    self.res_matrix[n_gpu][n_cpu][n_bw] = [-1, None, None]
