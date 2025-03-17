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

BASE_DIR = "./profile/vit/"

# model config
PARAMETER_SIZE = 85952456 / 1024 / 1024
BATCH_SIZE = 64
SEQ_HID = 768 * 512 * 12

ENV_PARAMS = tp.EnvParams(64, 34.057018390132875)


def fit_ViT():
    config_file = BASE_DIR + "config.csv"
    data_file = BASE_DIR + "data.csv"
    profile = load_profile_from_files(config_file, data_file)

    # profile
    num_nodes = np.array([key[0] for key in profile])
    num_gpus = np.array([key[1] for key in profile])
    parameter_size = np.array([key[2] for key in profile])
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
        parameter_size,
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

    profile_fwd_time = 40

    predict_outcome = {}

    # execution_plan: DP
    stParams = tp.StrategyParams(1, 1, 1, 0, 0, 0)
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
    actual_result = [153.97, 88, 62, 49]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["DP"] = predict_err

    # execution_plan: ZeRO-DP
    stParams = tp.StrategyParams(1, 1, 1, 0, 0, 0)
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
    actual_result = [144.93, 85, 65.97, 51]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-DP+GA"] = predict_err

    # execution_plan: ZeRO-Offload
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
    actual_result = [233, 139, 99, 83.99]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["ZeRO-Offload"] = predict_err

    # execution_plan: GC
    stParams = tp.StrategyParams(2, 1, 1, 0, 0, 0)
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
    actual_result = [195, 112, 73, 54.005]
    predict_err = [
        abs(predict - actual) / actual
        for predict, actual in zip(predict_result, actual_result)
    ]
    predict_outcome["GC"] = predict_err
    return predict_outcome


if __name__ == "__main__":
    fittable_params = fit_ViT()
    print("fittable parameters for ViT:", fittable_params)
    predict_outcome = validate(fittable_params)
    execution_plans_small = ["DP", "GC", "ZeRO-DP+GA", "ZeRO-Offload"]
    print("predictions:")
    for execution_plan in execution_plans_small:
        print(
            execution_plan,
            predict_outcome[execution_plan],
            "avg.:",
            round(np.average(predict_outcome[execution_plan]) * 100, 3),
            "max.:",
            round(np.max(predict_outcome[execution_plan]) * 100, 3),
        )
