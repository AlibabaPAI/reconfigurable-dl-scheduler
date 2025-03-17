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


BASE_DIR = "./profile/t5/"

# model config
PARAMETER_SIZE = 1240809472 / 1024 / 1024
BATCH_SIZE = 16
SEQ_HID = 1024 * 512 * 32

ENV_PARAMS = tp.EnvParams(64.0, 88.56587629910638)


def fit_T5():
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
    is_mp = np.array([key[7] for key in profile])
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
        is_mp,
        is_pp,
        str_params,
        ENV_PARAMS,
    )
    return fittable_params


def validate(fittable_params, history_data=True):
    perfParams = tp.PerfParams(*fittable_params)
    envParams = tp.EnvParams(*ENV_PARAMS)
    throughput = tp.ThroughputFunction(perfParams, envParams)

    profile_fwd_time = 338

    predict_outcome = {}

    # execution_plan: DP+TP+PP
    res_1818 = throughput.throughput_3D(
        PARAMETER_SIZE, 134, 1, 8, 1, 8, 16, 1024 * 512, 32, True
    )
    res_2828 = throughput.throughput_3D(
        PARAMETER_SIZE, 163, 2, 8, 2, 8, 16, 1024 * 512, 32, True
    )
    res_4848 = throughput.throughput_3D(
        PARAMETER_SIZE, 197, 4, 8, 4, 8, 16, 1024 * 512, 32, True
    )
    res_4444 = throughput.throughput_3D(
        PARAMETER_SIZE, 285, 4, 4, 4, 4, 16, 1024 * 512, 32, True
    )
    res_8282 = throughput.throughput_3D(
        PARAMETER_SIZE, 497, 8, 2, 8, 2, 16, 1024 * 512, 32, True
    )
    predict_result = [res_1818, res_2828, res_4848, res_4444, res_8282]
    actual_result = [615, 710, 815, 976.9, 1711]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["DP+TP+PP"] = predict_err

    # execution_plan: TP+PP
    res_2828 = throughput.throughput_3D(
        PARAMETER_SIZE, 163, 2, 8, 2, 8, 16, 1024 * 512, 32, True
    )
    res_4444 = throughput.throughput_3D(
        PARAMETER_SIZE, 285, 4, 4, 4, 4, 16, 1024 * 512, 32, True
    )
    res_8282 = throughput.throughput_3D(
        PARAMETER_SIZE, 497, 8, 2, 8, 2, 16, 1024 * 512, 32, True
    )
    predict_result = [res_2828, res_4444, res_8282]
    actual_result = [658.04, 1064, 1712.5]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["TP+PP"] = predict_err

    # execution_plan: ZeRO-DP+GA
    stParams = tp.StrategyParams(1, 1, 1, 0, 0, 2)
    result_1 = throughput.throughput(
        PARAMETER_SIZE, profile_fwd_time, 1, BATCH_SIZE, 1, 0, SEQ_HID, stParams, True
    )
    result_2 = throughput.throughput(
        PARAMETER_SIZE,
        profile_fwd_time / 2,
        2,
        BATCH_SIZE,
        1,
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
    actual_result = [1132.8, 721, 536, 463]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-DP+GA"] = predict_err

    # execution_plan: ZeRO-Offload+GC
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
    actual_result = [2411, 1431, 886, 653]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-Offload+GC"] = predict_err
    return predict_outcome


if __name__ == "__main__":
    fittable_params = fit_T5()
    print("fittable parameters for T5:", fittable_params)
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
