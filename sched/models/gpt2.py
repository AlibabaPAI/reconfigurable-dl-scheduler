import numpy as np
from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class gpt2(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 1920
        self.num_attention_heads = 19
        self.num_hidden_layers = 24
        self.sequence_length = 1024
        self.local_bsz = 16
        self.profile = {}
        self.orig_tpt = 16 / 0.62556

        self.env_params = EnvParams(
            B_pcie=59.720964902105614,
            B_nvlink=144.83763732303515,
        )
        self.perf_params = PerfParams(
            k_bwd=2.2955055547729417,
            k_sync=71.21658539688131,
            k_opt=0.05217141753878825,
            k_os_offload=4.877226916846699,
            k_const=1.786485024931732,
            k_off=0.027912731457899787,
            k_swap=51.48938818523304,
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
            "151": [5],
            "251": [55],
            "351": [555],
            "411": [1111],
            "451": [5555],
            "611": [111111],
            "651": [555555],
            "811": [11111111],
            "851": [55555555],
        }
        self.gpu_to_placements = {}

        self.placements_to_plan = {}
