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
BASE_DIR = "./profile/llama7b/"

# model config
PARAMETER_SIZE = 6738423808 / 1024 / 1024
BATCH_SIZE = 8
SEQ_HID = 4096 * 2048 * 32

ENV_PARAMS = tp.EnvParams(0.96, 150)


def fit_LLaMA7B():
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
        perfParams = load_perf_params_from_csv(HISTORY_DIR, "llama7B")
    else:
        perfParams = tp.PerfParams(*fittable_params)
    envParams = tp.EnvParams(*ENV_PARAMS)
    throughput = tp.ThroughputFunction(perfParams, envParams)

    predict_outcome = {}

    # execution_plan: DP+TP+PP
    res_1828 = throughput.throughput_3D(
        PARAMETER_SIZE, 300, 1, 8, 2, 8, 8, 4096 * 2048, 32, True
    )
    res_1848 = throughput.throughput_3D(
        PARAMETER_SIZE, 270, 1, 8, 4, 8, 8, 4096 * 2048, 32, True
    )
    res_1888 = throughput.throughput_3D(
        PARAMETER_SIZE, 243, 1, 8, 8, 8, 8, 4096 * 2048, 32, True
    )
    res_2444 = throughput.throughput_3D(
        PARAMETER_SIZE, 534, 2, 4, 4, 4, 8, 4096 * 2048, 32, True
    )
    predict_result = [res_1828, res_1848, res_1888, res_2444]
    actual_result = [1130, 945, 859, 1648]

    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["DP+TP+PP"] = predict_err

    # execution_plan: TP+PP
    res_2828 = throughput.throughput_3D(
        PARAMETER_SIZE, 380, 2, 8, 2, 8, 8, 4096 * 2048, 32, True
    )
    res_8888 = throughput.throughput_3D(
        PARAMETER_SIZE, 494, 8, 8, 8, 8, 8, 4096 * 2048, 32, True
    )
    res_4848 = throughput.throughput_3D(
        PARAMETER_SIZE, 433, 4, 8, 4, 8, 8, 4096 * 2048, 32, True
    )
    res_4444 = throughput.throughput_3D(
        PARAMETER_SIZE, 740, 4, 4, 4, 4, 8, 4096 * 2048, 32, True
    )
    res_8282 = throughput.throughput_3D(
        PARAMETER_SIZE, 1435, 8, 2, 8, 2, 8, 4096 * 2048, 32, True
    )
    predict_result = [
        res_2828,
        res_8888,
        res_4848,
        res_4444,
        res_8282,
    ]
    actual_result = [1569, 1871, 1665.7, 2641, 4817]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["TP+PP"] = predict_err

    # execution_plan: ZeRO-Offload+GC
    profile_fwd_time = 3

    # # Offload
    stParams = tp.StrategyParams(2, 1, 1, 0, 0, 1)
    result_1 = throughput.throughput(
        PARAMETER_SIZE, profile_fwd_time, 1, BATCH_SIZE, 4, 0, SEQ_HID, stParams, True
    )
    result_2 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 2,
        2,
        BATCH_SIZE,
        4,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    result_3 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 4,
        4,
        BATCH_SIZE,
        4,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    result_4 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 8,
        8,
        BATCH_SIZE,
        4,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    predict_result = [result_1, result_2, result_3, result_4]
    actual_result = [57879, 30072, 16098, 9400]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-Offload+GC"] = predict_err
    return predict_outcome


if __name__ == "__main__":
    fittable_params = fit_LLaMA7B()
    print("fittable parameters for LLaMA-2-7B:", fittable_params)
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
