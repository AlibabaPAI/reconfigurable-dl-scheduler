import numpy as np
from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class llama7(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_hidden_layers = 32
        self.sequence_length = 2048
        self.local_bsz = 8
        self.orig_tpt = 8 / 9.4

        self.profile = {}
        self.env_params = EnvParams(
            B_pcie=0.96,
            B_nvlink=150,
        )
        self.perf_params = PerfParams(
            k_bwd=2.0,
            k_sync=10.0,
            k_opt=0.03,
            k_os_offload=1e-08,
            k_const=1.0492757598487223,
            k_off=1.1192066928294584,
            k_swap=1000.0,
        )

        self.max_dp_atom_bsz = {
            "DP": {"gc": 0, "other": 0},
            "zero-dp": {"gc": 0, "other": 2},
            "zero-offload": {"gc": 0, "other": 8},
        }
        self.max_mp_atom_bsz = {"other": [0] + [8] * 63, "gc": [0] + [8] * 63}
        self.cpu_strategy = {
            "zero-offload": 8,
        }
        self.mem_strategy = {
            "zero-offload": 5.1,
        }

        self.sspeed_strategy = {
            "zero-offload": 5.1,
        }
        self.synergy_speedup = 1
        self.avail_placement = {
            "248": [88888888],
            "284": [88888888],
            "428": [88888888],
            "444": [88888888],
            "188": [88888888],
            "184": [8888],
            "228": [8888, 44444444],
            "482": [88888888],
            "148": [8888, 44444444],
            "282": [8888],
            "182": [88],
            "144": [88, 4444],
            "244": [8888, 44444444],
            "818": [88888888],
            "424": [8888, 44444444],
            "418": [8888, 44444444],
            "824": [88888888],
            "224": [88, 4444, 22222222],
            "242": [88, 4444],
            "442": [8888, 44444444],
            "842": [88888888],
            "281": [88],
            "481": [8888],
            "881": [88888888],
            "814": [44444444],
            "414": [4444, 22222222],
            "422": [4444, 22222222],
            "822": [44444444],
            "441": [4444],
            "841": [44444444],
            "812": [22222222],
            "821": [22222222],
        }

        self.gpu_to_placements = {
            16: [88, 4444, 22222222],
            64: [88888888],
            32: [8888, 44444444],
            8: [8],
            4: [4],
            2: [2],
            1: [1],
        }

        self.placements_to_plan = {
            88888888: "248",
            8888: "244",
            88: "182",
            44444444: "244",
            4444: "242",
            22222222: "224",
            8: "zero-offload",
            4: "zero-offload",
            2: "zero-offload",
            1: "zero-offload",
        }
