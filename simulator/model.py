from models import *
from profile_forward import build_model
from simulator.model_solver import ModelSolver
import numpy as np
from pred_throughput import fit_perf_params


class Model():
    def __init__(self, name):
        self.model_name = name
        model = globals()[self.model_name+"_1"](self.model_name)
        self.local_bsz = model.local_bsz
        self.forward_time, self.model_params = build_model(
            self.model_name, model.hidden_size, model.num_attention_heads, model.num_hidden_layers, 1)
        self.seq_hid = model.hidden_size*model.sequence_length*model.num_hidden_layers
        self.jobSolver = ModelSolver(self.model_name, self.model_params, self.local_bsz,
                                     self.forward_time, self.seq_hid, model.max_ddp_atom_bsz, model.max_mp_atom_bsz)
        self.cpu_strategy = model.cpu_strategy
        self.mem_strategy = model.mem_strategy
        self.sspeed_strategy = model.sspeed_strategy
        self.synergy_speedup = model.synergy_speedup
        self.profile = model.profile
        self.env_para = 6.890823514935245
        self.jobSolver.perf_params = model.perfParams

    def update_params(self):
        num_nodes = np.array([key[0] for key in self.profile])
        num_gpus = np.array([key[1] for key in self.profile])
        ps = np.array([key[2] for key in self.profile])
        batch_size = np.array([key[3] for key in self.profile])
        seq_hid = np.array([key[4] for key in self.profile])
        num_threads = np.array([key[5] for key in self.profile])
        zero_type = np.array([key[6] for key in self.profile])
        is_mp = np.array([key[7] for key in self.profile])
        is_pp = np.array([key[8] for key in self.profile])

        forward_time = np.array([val[0] for val in self.profile.values()])
        overlap_sync = np.array([val[1] for val in self.profile.values()])
        optimize_time = np.array([val[2] for val in self.profile.values()])
        iteration_time = np.array([val[3] for val in self.profile.values()])
        str_params = np.array([val[4] for val in self.profile.values()])
        self.jobSolver.perf_params = fit_perf_params(self.env_para,
                                                     num_nodes, num_gpus, ps, batch_size,
                                                     forward_time, overlap_sync,
                                                     optimize_time, iteration_time,
                                                     seq_hid, num_threads,
                                                     zero_type, is_mp, is_pp, str_params)
        return self.jobSolver.perf_params


MODELS = {
    "gpt2": Model("gpt2"),
    "bert": Model("bert"),
    "vit": Model("vit"),
    "t5": Model("t5"),
    "roberta": Model("roberta"),
    "swin": Model("swin")
}
