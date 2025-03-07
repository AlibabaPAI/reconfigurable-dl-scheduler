import numpy as np
from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class llama30(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 1536
        self.num_attention_heads = 16
        self.num_hidden_layers = 24
        self.sequence_length = 1024
        self.local_bsz = 4
        self.orig_tpt = 4 / 2.759

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

        self.avail_placement = {
            "444": [88888888],
            "344": [888888, 66666666],
            "624": [888888],
            "524": [88888, 55555555],
            "542": [88888],
            "642": [888888, 66666666],
            "442": [8888],
            "641": [444444],
        }

        self.gpu_to_placements = {
            48: [888888],
            40: [88888],
            32: [8888],
            24: [444444],
        }

        self.placements_to_plan = {
            888888: "344",
            88888: "524",
            55555555: "524",
            8888: "442",
            66666666: "344",
            444444: "641",
        }
