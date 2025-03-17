import importlib
from morphling_sched.profile_forward import build_model
from morphling_sched.performance_model import PerformanceModel
import numpy as np
from morphling_sched.pred_throughput import fit_perf_params


class Model:
    def __init__(self, name):
        self.model_name = name
        try:
            module = importlib.import_module(f"models.{self.model_name}")
            model_class = getattr(module, self.model_name)
            model = model_class(self.model_name)
        except Exception as e:
            print("error raised", e)

        self.local_bsz = model.local_bsz
        self.forward_time, self.model_flops, self.parameter_size = build_model(
            self.model_name,
            model.hidden_size,
            model.num_attention_heads,
            model.num_hidden_layers,
            1,
        )
        seq_hid = model.hidden_size * model.sequence_length * model.num_hidden_layers
        self.model_solver = PerformanceModel(
            self.model_name,
            self.parameter_size,
            self.local_bsz,
            self.forward_time,
            seq_hid,
            model.num_hidden_layers,
            model.max_dp_atom_bsz,
            model.max_mp_atom_bsz,
            model.avail_placement,
            model.env_params,
            model.perf_params,
        )
        self.cpu_strategy = model.cpu_strategy
        self.mem_strategy = model.mem_strategy
        self.sspeed_strategy = model.sspeed_strategy
        self.synergy_speedup = model.synergy_speedup
        self.profile = model.profile
        self.env_para = model.env_params
        # large model
        self.avail_placement = getattr(model, "avail_placement", None)
        self.placements_to_plan = getattr(model, "placements_to_plan", None)
        self.gpu_to_placements = getattr(model, "gpu_to_placements", None)
        self.model_solver.perf_params = model.perf_params
        self.orig_tpt = model.orig_tpt

    def update_params(self):
        num_nodes = np.array([key[0] for key in self.profile])
        num_gpus = np.array([key[1] for key in self.profile])
        ps = np.array([key[2] for key in self.profile])
        batch_size = np.array([key[3] for key in self.profile])
        seq_hid = np.array([key[4] for key in self.profile])
        num_threads = np.array([key[5] for key in self.profile])
        zero_type = np.array([key[6] for key in self.profile])
        is_mp = np.array([key[7] for key in self.profile])

        forward_time = np.array([val[0] for val in self.profile.values()])
        overlap_sync = np.array([val[1] for val in self.profile.values()])
        optimize_time = np.array([val[2] for val in self.profile.values()])
        iteration_time = np.array([val[3] for val in self.profile.values()])
        str_params = np.array([val[4] for val in self.profile.values()])
        self.model_solver.perf_params = fit_perf_params(
            self.env_para,
            num_nodes,
            num_gpus,
            ps,
            batch_size,
            forward_time,
            overlap_sync,
            optimize_time,
            iteration_time,
            seq_hid,
            num_threads,
            zero_type,
            is_mp,
            str_params,
        )
        return self.model_solver.perf_params


MODELS = {
    "gpt2": Model("gpt2"),
    "bert": Model("bert"),
    "vit": Model("vit"),
    "t5": Model("t5"),
    "roberta": Model("roberta"),
    "llama7": Model("llama7"),
    "llama30": Model("llama30"),
}
