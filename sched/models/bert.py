import numpy as np
from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class bert(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 1536
        self.num_attention_heads = 16
        self.num_hidden_layers = 24
        self.sequence_length = 1024
        self.local_bsz = 32
        self.profile = {}
        self.orig_tpt = 32 / 0.15768
        self.env_params = EnvParams(
            B_pcie=64,
            B_nvlink=119.4879085392442,
        )
        self.perf_params = PerfParams(
            k_bwd=1.7762507345127174,
            k_sync=1.1410390978600897,
            k_opt=1e-08,
            k_os_offload=4.400603962116795,
            k_const=15.257464643432971,
            k_off=37.288983000053264,
            k_swap=26.221881110226107,
        )

        self.max_dp_atom_bsz = {
            "DP": {"gc": 32, "other": 32},
            "zero-dp": {"gc": 0, "other": 32},
            "zero-offload": {"gc": 0, "other": 32},
        }
        self.max_mp_atom_bsz = {"other": [0] + [8] * 63, "gc": [0] + [8] * 63}
        self.cpu_strategy = {
            "3D": 1,
            "ga": 1,
            "gc": 1,
            "zero-dp": 1,
            "zero-offload": 8,
        }
        self.mem_strategy = {
            "3D": 1.2,
            "ga": 1.2,
            "gc": 1.2,
            "zero-dp": 1.5,
            "zero-offload": 5.1,
        }

        self.sspeed_strategy = {
            "3D": 1.2,
            "ga": 1.2,
            "gc": 1.2,
            "zero-dp": 1.5,
            "zero-offload": 5.1,
        }
        self.synergy_speedup = 1
        self.avail_placement = {}
