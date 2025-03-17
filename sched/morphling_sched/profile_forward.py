CUDA_ENV = False
if CUDA_ENV:
    import torch
    import deepspeed.profiling.flops_profiler as flops

    import numpy as np
    import h5py
    import time
    import deepspeed.profiling.flops_profiler as flops
    from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
    from collections.abc import Mapping
    from transformers import (
        AutoModelForMaskedLM,
        CONFIG_MAPPING,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )


def prepare_input(data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=torch.device("cuda:%d" % 0))
        return data.to(**kwargs)
    return data


def prepare_inputs(inputs):
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    inputs = prepare_input(inputs)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
        )

    return inputs


def model_forward(model, inputs):
    outputs = model(**inputs)


def run_forward(model, inputs):
    forward_time = []
    for i in range(0, 10):
        start_time = time.time()
        model_forward(model, inputs)
        if i > 3:
            forward_time.append(time.time() - start_time)
    return np.mean(forward_time)


def build_model(
    model_type, hidden_size, num_attention_heads, hidden_layers, atomic_bsz
):
    if not CUDA_ENV:
        if model_type == "roberta":
            return 480, 0, 355412057 / 1024 / 1024
        elif model_type == "vit":
            return 40, 0, 85952456 / 1024 / 1024
        elif model_type == "bert":
            return 225, 0, 336297858 / 1024 / 1024
        elif model_type == "gpt2":
            return 1100, 0, 1557686400 / 1024 / 1024
        elif model_type == "t5":
            return 410, 0, 1240809472 / 1024 / 1024
        else:
            return 1000, 3343242, 6738423808 / 1024 / 1024
    else:
        config = CONFIG_MAPPING[model_type]()
        config.update_from_string(
            "hidden_size=%s,num_attention_heads=%s,num_hidden_layers=%s"
            % (hidden_size, num_attention_heads, hidden_layers)
        )
        model = AutoModelForCausalLM.from_config(config)
        cuda_condition = torch.cuda.is_available()
        device = torch.device("cuda:%d" % 0 if cuda_condition else "cpu")
        model.to(device)
        prof = flops.FlopsProfiler(model)
        inputs = {
            "input_ids": torch.randint(1, 2, [atomic_bsz, 1024]),
            "attention_mask": torch.ones(atomic_bsz, 1024),
            "labels": torch.randint(1, 2, [atomic_bsz, 1024]),
        }
        inputs = prepare_inputs(inputs)
        prof.start_profile()
        mean_forward_time = run_forward(model, inputs)
        model_flops = prof.get_total_flops()
        params = prof.get_total_params()
        prof.end_profile()
        return mean_forward_time, model_flops, params
