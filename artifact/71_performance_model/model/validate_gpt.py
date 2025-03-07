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
from utils import load_profile_from_files

import numpy as np


BASE_DIR = "./profile/gpt2/"

# model config
PARAMETER_SIZE = 1557686400 / 1024 / 1024
BATCH_SIZE = 16
SEQ_HID = 1600 * 1024 * 48

ENV_PARAMS = tp.EnvParams(59.720964902105614, 144.837637323036)


def fit_GPT2():
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


def validate(fittable_params, history_data=False):
    perfParams = tp.PerfParams(*fittable_params)
    envParams = tp.EnvParams(*ENV_PARAMS)
    throughput = tp.ThroughputFunction(perfParams, envParams)

    profile_fwd_time = 1100

    predict_outcome = {}

    # execution_plan: DP+TP+PP
    res_151 = throughput.throughput_3D(
        PARAMETER_SIZE, 290, 1, 5, 1, 5, 16, 1600 * 1024, 24, True
    )
    res_351 = throughput.throughput_3D(
        PARAMETER_SIZE, 290, 3, 5, 3, 5, 16, 1600 * 1024, 24, True
    )
    res_251 = throughput.throughput_3D(
        PARAMETER_SIZE, 290, 2, 5, 2, 5, 16, 1600 * 1024, 24, True
    )
    res_451 = throughput.throughput_3D(
        PARAMETER_SIZE, 290, 4, 5, 4, 5, 16, 1600 * 1024, 24, True
    )
    res_651 = throughput.throughput_3D(
        PARAMETER_SIZE, 290, 6, 5, 6, 5, 16, 1600 * 1024, 24, True
    )
    predict_result = [res_151, res_351, res_251, res_451, res_651]
    actual_result = [1052.5, 1066, 1080.7, 1070, 1078]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["DP+TP+PP"] = predict_err

    # execution_plan: TP+PP
    res_451 = throughput.throughput_3D(
        PARAMETER_SIZE, 290, 4, 5, 4, 5, 16, 1600 * 1024, 24, True
    )
    res_411 = throughput.throughput_3D(
        PARAMETER_SIZE, 920, 4, 1, 4, 1, 16, 1600 * 1024, 24, True
    )
    res_811 = throughput.throughput_3D(
        PARAMETER_SIZE, 920, 8, 1, 8, 1, 16, 1600 * 1024, 24, True
    )
    predict_result = [res_451, res_411, res_811]
    actual_result = [1068, 3089.3, 3120]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["TP+PP"] = predict_err

    # execution_plan: ZeRO-DP+GA
    stParams = tp.StrategyParams(8, 8, 8, 0, 0, 2)
    result_1 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 8,
        1,
        BATCH_SIZE,
        1,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    stParams = tp.StrategyParams(4, 4, 4, 0, 0, 2)
    result_2 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 8,
        2,
        BATCH_SIZE,
        1,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    stParams = tp.StrategyParams(1, 1, 1, 0, 0, 2)
    result_3 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 4,
        4,
        BATCH_SIZE,
        1,
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
        1,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    predict_result = [result_1, result_2, result_3, result_4]
    actual_result = [3617, 1880, 982, 496.76]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-DP+GA"] = predict_err

    stParams = tp.StrategyParams(1, 1, 1, 0, 0, 1)
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
        profile_fwd_time / 4,
        8,
        BATCH_SIZE,
        4,
        0,
        SEQ_HID,
        stParams,
        True,
    )
    predict_result = [result_1, result_2, result_3, result_4]
    actual_result = [5230, 2995, 1567.3, 1203]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-Offload+GC"] = predict_err
    return predict_outcome


if __name__ == "__main__":
    fittable_params = fit_GPT2()
    print("fittable parameters for GPT-2:", fittable_params)
    predict_outcome = validate(fittable_params)
    execution_plans_large = ["TP+PP", "DP+TP+PP", "ZeRO-DP+GA", "ZeRO-Offload+GC"]
    print("predictions:")
    for execution_plan in execution_plans_large:
        print(
            execution_plan,
            predict_outcome[execution_plan],
            "avg.:",
            round(np.average(predict_outcome[execution_plan]) * 100, 3),
            "max.:",
            round(np.max(predict_outcome[execution_plan]) * 100, 3),
        )
