import numpy as np
from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class roberta(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.num_hidden_layers = 24
        self.sequence_length = 1024
        self.local_bsz = 64
        self.orig_tpt = 64 / 0.234
        self.profile = {}
        self.max_dp_atom_bsz = {
            "DP": {"gc": 64, "other": 64},
            "zero-dp": {"gc": 0, "other": 64},
            "zero-offload": {"gc": 0, "other": 64},
        }
        self.max_mp_atom_bsz = {"other": [32] * 64, "gc": [16] * 64}
        self.env_params = EnvParams(
            B_pcie=7.9107385391936935,
            B_nvlink=50.303643979067175,
        )
        self.perf_params = PerfParams(
            k_bwd=1.7226216091452244,
            k_sync=1.205493103963666,
            k_opt=1e-08,
            k_os_offload=2.200462132207284,
            k_const=15.73715000699807,
            k_off=16.635258404842535,
            k_swap=32.9530767089104,
        )
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
