import numpy as np
from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class t5(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 1280
        self.num_attention_heads = 16
        self.num_hidden_layers = 16
        self.sequence_length = 768
        self.local_bsz = 16
        self.orig_tpt = 16 / 0.3207
        self.profile = {}
        self.env_params = EnvParams(
            B_pcie=64.0,
            B_nvlink=161.02246829023724,
        )
        self.perf_params = PerfParams(
            k_bwd=1.2754324551852874,
            k_sync=1.0,
            k_opt=1e-08,
            k_os_offload=3.4538198406863163,
            k_const=11.616030890964971,
            k_off=224.86968171631415,
            k_swap=61.68023191529816,
        )
        self.max_dp_atom_bsz = {
            "DP": {"gc": 16, "other": 16},
            "zero-dp": {"gc": 0, "other": 16},
            "zero-offload": {"gc": 0, "other": 16},
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
            "3D": 1.5,
            "ga": 1.5,
            "gc": 1.5,
            "zero-dp": 8.1,
            "zero-offload": 1.5,
        }

        self.sspeed_strategy = {
            "3D": 1.5,
            "ga": 1.5,
            "gc": 1.5,
            "zero-dp": 8.1,
            "zero-offload": 1.5,
        }
        self.synergy_speedup = 2
        self.avail_placement = {
            "141": [4],
            "181": [8],
            "221": [22],
            "241": [44],
            "281": [88],
            "321": [222],
            "341": [444],
            "381": [888],
            "411": [1111],
            "421": [2222],
            "441": [4444],
            "481": [8888],
            "611": [111111],
            "621": [222222],
            "641": [444444],
            "681": [888888],
            "811": [11111111],
            "821": [22222222],
            "841": [44444444],
            "881": [88888888],
        }

        self.gpu_to_placements = {}

        self.placements_to_plan = {}
