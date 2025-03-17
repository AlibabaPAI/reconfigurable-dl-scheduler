import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sched_dir = os.path.abspath(os.path.join(current_dir, "../../sched"))
if sched_dir not in sys.path:
    sys.path.append(sched_dir)

os.chdir(current_dir)
from morphling_sched.pred_throughput import (
    ThroughputFunction,
    StrategyParams,
    PerfParams,
    EnvParams,
)


class t5_3B:
    def __init__(self):
        return

    def model_info(self):
        self.hidden_size = 2048
        self.num_attention_heads = 24
        self.num_hidden_layers = 12
        self.sequence_length = 2048
        self.local_bsz = 4
        self.orig_tpt = 8 / 9.4

        self.profile = {}
        self.env_params = EnvParams(
            B_nvlink=150,
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
            "zero-offload": {"gc": 0, "other": 4},
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
            "131": [3],
            "121": [2],
            "141": [4],
        }
        self.gpu_to_placements = {
            3: [3],
            2: [2],
            4: [4],
        }

        self.placements_to_plan = {
            3: "131",
            2: "121",
            4: "141",
        }
