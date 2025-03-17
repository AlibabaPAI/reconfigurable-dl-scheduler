import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
simulator_dir = os.path.abspath(
    os.path.join(current_dir, "../../../sched/morphling_sched")
)
if simulator_dir not in sys.path:
    sys.path.append(simulator_dir)

os.chdir(current_dir)

import pred_throughput as tp
from utils import load_profile_from_files, load_perf_params_from_csv

import numpy as np

HISTORY_DIR = "./profile/history.csv"
BASE_DIR = "./profile/llama30b/"

# model config
PARAMETER_SIZE = 33170309120 / 1024 / 1024
BATCH_SIZE = 4
SEQ_HID = 5120 * 2048 * 40

ENV_PARAMS = tp.EnvParams(0.96, 150)


def fit_LLaMA30B():
    config_file = BASE_DIR + "config.csv"
    data_file = BASE_DIR + "data.csv"
    profile = load_profile_from_files(config_file, data_file)

    # profile
    num_nodes = np.array([key[0] for key in profile])
    num_gpus = np.array([key[1] for key in profile])
    ps = np.array([key[2] for key in profile])
    batch_size = np.array([key[3] for key in profile])
    seq_hid = np.array([key[4] for key in profile])
    num_threads = np.array([key[5] for key in profile])
    zero_type = np.array([key[6] for key in profile])
    is_tp = np.array([key[7] for key in profile])
    is_pp = np.array([key[8] for key in profile])

    forward_time = np.array([val[0] for val in profile.values()])
    overlap_sync = np.array([val[1] for val in profile.values()])
    optimize_time = np.array([val[2] for val in profile.values()])
    iteration_time = np.array([val[3] for val in profile.values()])
    str_params = np.array([val[4] for val in profile.values()])

    fittable_params = tp.fit_perf_params(
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
        is_tp,
        is_pp,
        str_params,
        ENV_PARAMS,
    )
    return fittable_params


def validate(fittable_params, history_data=True):
    if history_data:
        perfParams = load_perf_params_from_csv(HISTORY_DIR, "llama30B")
    else:
        perfParams = tp.PerfParams(*fittable_params)
    envParams = tp.EnvParams(*ENV_PARAMS)
    throughput = tp.ThroughputFunction(perfParams, envParams)

    predict_outcome = {}

    # execution_plan: DP+TP+PP
    res_5258 = throughput.throughput_3D(
        PARAMETER_SIZE, 800, 5, 2, 5, 8, 4, 5120 * 2048, 40, True
    )
    res_5458 = throughput.throughput_3D(
        PARAMETER_SIZE, 800, 5, 4, 5, 8, 4, 5120 * 2048, 40, True
    )
    res_4448 = throughput.throughput_3D(
        PARAMETER_SIZE, 750, 4, 4, 4, 4, 4, 5120 * 2048, 40, True
    )
    res_4488 = throughput.throughput_3D(
        PARAMETER_SIZE, 563, 4, 4, 8, 8, 4, 5120 * 2048, 40, True
    )
    predict_result = [res_5258, res_5458, res_4448, res_4488]
    actual_result = [2661, 2741, 2758, 2017]

    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["DP+TP+PP"] = predict_err

    # execution_plan: TP+PP
    res_6464 = throughput.throughput_3D(
        PARAMETER_SIZE, 1500, 6, 4, 6, 4, 4, 5120 * 2048, 40, True
    )
    res_6468 = throughput.throughput_3D(
        PARAMETER_SIZE, 850, 6, 4, 6, 8, 4, 5120 * 2048, 40, True
    )
    predict_result = [res_6464, res_6468]
    actual_result = [5296, 2741]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["TP+PP"] = predict_err
    return predict_outcome


if __name__ == "__main__":
    fittable_params = fit_LLaMA30B()
    print("fittable parameters for LLaMA-30B:", fittable_params)
    predict_outcome = validate(fittable_params)
    execution_plans_large = ["TP+PP", "DP+TP+PP", "ZeRO-DP+GA", "ZeRO-Offload+GC"]
    print("predictions:")
    for execution_plan in execution_plans_large:
        if execution_plan not in predict_outcome:
            continue
        print(
            execution_plan,
            predict_outcome[execution_plan],
            "avg.:",
            round(np.average(predict_outcome[execution_plan]) * 100, 3),
            "max.:",
            round(np.max(predict_outcome[execution_plan]) * 100, 3),
        )
