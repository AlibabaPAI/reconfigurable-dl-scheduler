import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
simulator_dir = os.path.abspath(os.path.join(current_dir, "../../sched"))
if simulator_dir not in sys.path:
    sys.path.append(simulator_dir)

os.chdir(current_dir)
from morphling_sched.pred_throughput import (
    ThroughputFunction,
    StrategyParams,
    PerfParams,
    EnvParams,
)


class llama7:
    def __init__(self):
        return

    def model_info(self):
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_hidden_layers = 32
        self.sequence_length = 2048
        self.local_bsz = 16
        self.orig_tpt = 8 / 9.4

        self.profile = {}
        self.env_params = EnvParams(
            B_nvlink=110,
            B_pcie=0.96,
        )
        self.perf_params = PerfParams(
            k_bwd=2,
            k_sync=10,
            k_opt=0.03,
            k_os_offload=4,
            k_const=1.049,
            k_off=1.119,
            k_swap=1000,
        )

        self.max_dp_atom_bsz = {
            "DP": {"gc": 0, "other": 0},
            "zero-dp": {"gc": 0, "other": 0},
            "zero-offload": {"gc": 16, "other": 0},
        }
        self.max_mp_atom_bsz = {"other": [0] + [0] * 63, "gc": [0] + [0] * 63}
        self.cpu_strategy = {
            "zero-offload": 8,
        }
        self.mem_strategy = {
            "zero-offload": 5.1,
        }

        self.sspeed_strategy = {
            "zero-offload": 5.1,
        }
        self.avail_placement = {
            "184": [8888],
            "228": [8888, 44444444],
            "148": [8888, 44444444],
            "144": [88, 4444],
            "244": [8888, 44444444],
            "418": [8888, 44444444],
            "224": [88, 4444, 22222222],
            "242": [88, 4444],
            "281": [88],
            "822": [44444444],
            "441": [4444],
            "841": [44444444],
            "221": [4],
            "141": [4],
        }
