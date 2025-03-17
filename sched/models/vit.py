from models.model_info import ModelInfo
from morphling_sched.pred_throughput import PerfParams, EnvParams


class vit(ModelInfo):
    def __init__(self, name):
        super().__init__(name, 1)

    def model_info(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.sequence_length = 1024
        self.local_bsz = 64
        self.profile = {}
        self.orig_tpt = 64 / 0.049219
        self.env_params = EnvParams(
            B_pcie=6.1646901221266655,
            B_nvlink=29.7451292741858,
        )

        self.perf_params = PerfParams(
            k_bwd=1.0,
            k_sync=13.54749257672695,
            k_opt=0.09786337059659733,
            k_os_offload=1e-08,
            k_const=5.201298157965654,
            k_off=15.692686524178942,
            k_swap=21.647413513688274,
        )
        self.max_dp_atom_bsz = {
            "DP": {"gc": 64, "other": 64},
            "zero-dp": {"gc": 0, "other": 64},
            "zero-offload": {"gc": 0, "other": 64},
        }
        self.max_mp_atom_bsz = {"other": [64] * 64, "gc": [64] * 64}
        self.cpu_strategy = {
            "3D": 1,
            "ga": 1,
            "gc": 1,
            "zero-dp": 1,
            "zero-offload": 8,
        }
        self.mem_strategy = {
            "3D": 2,
            "ga": 2,
            "gc": 1.9,
            "zero-dp": 2,
            "zero-offload": 3,
        }

        self.sspeed_strategy = {
            "3D": 2,
            "ga": 2,
            "gc": 1.9,
            "zero-dp": 2,
            "zero-offload": 3,
        }
        self.synergy_speedup = 1.2
        self.avail_placement = {}
