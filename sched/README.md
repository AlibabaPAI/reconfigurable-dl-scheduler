# Morphling Testbed Experiments

## Code Organization
- `benckmark` and `sched/` contains code for testbed experiments and is partially adapted from [Pollux](https://github.com/petuum/adaptdl/tree/osdi21-artifact).
    - `benchmark/models/` contains the implementations of each evaluated model described in Table 1.
    - `benchmark/workloads/` contains the job trace used to evaluate Morphling, Sia, Synergy and Antman.
    - `benchmark/run_workload.py` submits jobs according to a workload trace.
    - `benchmark/run_monitor.py` monitors and logs the cluster state during each cluster scheduling experiment.
    - `sched/morphling-sched/` contains testbed training code for testbed experiments.
    	- `sched/morphling-sched/policy/morphling.py` containes the implementation of Morphling scheduling algorithm.

## Testbed Setup

The testbed experiments require 8 nodes, each with 8 NVIDIA A800 GPUs (80 GB), 96 vCPUs, 1,600 GB memory, 400 GB/s NVLink bandwidth, and 100 GB/s RDMA network bandwidth. The cluster experiments in the paper are all completed by accessing the corporation internal GPU cluster, thus it is highly related to internal testbed platform and is hard to make them publicly available to the  AE reviewers.

Additionly, Cloud Parallel File Storage (CPFS) is required for dataset and transformer model checkpoint storage to speed up the I/O process. At least 800G  is needed on each node for storage.