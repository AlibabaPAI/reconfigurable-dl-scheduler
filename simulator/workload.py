import argparse
import numpy as np
import os
import pandas
import random
from datetime import datetime, timedelta
import math


def generate(num_jobs, start=0, duration=24, seed=0, cpu_ratio=8, bw_ratio=0.15625, dataset_type="base"):
    trace_csv = os.path.join(os.path.dirname(__file__),
                             "traces", "philly.csv")
    trace = pandas.read_csv(trace_csv, parse_dates=["submit_time"])
    st = 1508234400
    et = 1508277600
    trace = trace[((trace.timestamp >= st) & (trace.end_time < et)) | ((trace.timestamp >= st) & (
        trace.timestamp < et)) | ((trace.end_time >= st) & (trace.end_time < et))]
    num_workloads = len(trace)*64//200+1
    trace = trace[trace.GPUs <= 64]
    trace = trace[trace.GPUs >= 1]
    trace = trace[trace.GPUs != 7]
    trace = trace[trace.GPUs != 6]
    trace = trace[trace.runtime >= 15]
    trace_mutli_gpu = trace[(trace.GPUs*trace.runtime >
                             3600) | (trace.GPUs > 1)]
    trace_single_gpu = trace[(trace.GPUs*trace.runtime <= 3600)]
    rng = random.Random(1234)
    sample_multi = trace_mutli_gpu.sample(
        n=int(55), random_state=rng.randint(0, 1 << 32))
    sample_single = trace_single_gpu.sample(
        n=num_workloads-int(55), random_state=rng.randint(0, 1 << 32))
    records = []
    avg_runtime = []
    load = 0
    for sample in [sample_single, sample_multi]:
        for row in sample.itertuples():
            rec = {}
            rec["submit_time"] = row.timestamp
            rec["runtime"] = min(row.runtime, 3600)
            avg_runtime.append(rec["runtime"])
            num_gpus = min(row.GPUs, 8)
            if row.GPUs*row.runtime > 3600 or row.GPUs > 1:
                num_gpus = max(2, num_gpus)
                rec["application"] = rng.choice(["gpt2", "t5", "bert"])
            else:
                num_gpus = 1
                rec["application"] = rng.choice(["vit", "swin", "roberta"])
            rec["num_gpus"] = num_gpus
            rec["num_cpus"] = num_gpus*cpu_ratio
            if num_gpus > 8:
                rec["bandwidth"] = num_gpus/math.ceil(num_gpus/8)*bw_ratio
            else:
                rec["bandwidth"] = 0
            load += num_gpus*rec["runtime"]
            rec["strategy"] = rng.choice(
                ["3D"]*5+["gc", "zero-dp", "zero-offload"])
            if rec["strategy"] == "zero-dp":
                if rec["application"] == "bert" and rec["num_gpus"] == 2:
                    rec["strategy"] = "zero-offload"
                if rec["application"] == "gpt2" and rec["num_gpus"] == 2:
                    rec["strategy"] = "zero-offload"
            records.append(rec)
    records.sort(key=lambda v: v["submit_time"])
    min_sub = min([rsc["submit_time"] for rsc in records])
    for i in range(0, len(records)):
        records[i]["submit_time"] -= min_sub
    for idx, rec in enumerate(records):
        rec["name"] = "{}-{}".format(rec["application"], idx)
    return pandas.DataFrame(records, columns=("name", "submit_time", "runtime", "application",
                                              "num_gpus", "num_cpus", "bandwidth", "strategy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=0,
                        help="starting hour")
    parser.add_argument("-d", "--duration", type=int, default=24,
                        help="total number of workload hours")
    parser.add_argument("-n", "--num-jobs", type=int, default=1000,
                        help="total number of jobs")
    parser.add_argument("-o", "--output", type=str, default="model_2x.csv",
                        help="path to output the workload")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--cpu-ratio", type=int, default=8,
                        help="ratio of cpu and gpu",)
    parser.add_argument("--bw-ratio", type=int, default=0.15625,
                        help="ratio of bandwidth and gpu")
    parser.add_argument("--dataset-tyoe", type=str,
                        default="base", choices=["base", "sia-biased"])
    args = parser.parse_args()
    workload = generate(args.num_jobs, start=args.start,
                        duration=args.duration, seed=args.seed, cpu_ratio=args.cpu_ratio, bw_ratio=args.bw_ratio, dataset_type=args.dataset_type)
    csv = workload.set_index("name").to_csv(args.output)
    print(workload.groupby(["application"])
          .size().reset_index(name="count"))
