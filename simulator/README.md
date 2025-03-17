# Artifact for Morphling (MLSys'25)

## Overview
This is the artifact for the MLSys'25 paper "Morphling: Exploiting Job Reconfigurability for Deep Learning Cluster Scheduling".

## Why We Use the Simulator

Using a real cluster (with 64 A800 GPUs) is both time-consuming and costly. To address this, we pre-collected performance data for all the models listed in Table 1 under various resource limits and execution plans, and the data is saved at `./simulator/traces/`. In this way, simulator can bypass the actual runtime of the jobs, thereby enabling us to focus more on optimizing the scheduling system.

Note that the simulator only replaces the physical execution of the computational resources (i.e., GPUs); the core components of the system, such as the performance model and scheduling algorithm in Morphling, still function as originally designed.

## Fidelity of Simulator

For the performance model validation and micro-benchmarks, the number of resources and models involved is relatively small. As a result, the simulator results are nearly identical to those reported in the paper, with very small errors compared to real GPU experiments.

However, for cluster experiments, the longer runtime introduces unavoidable factors that can affect the results, such as network fluctuations and restart delays. Although these factors have been accounted for in the simulator, it is impossible to precisely predict their impact. Therefore, compared to real-world running, the simulator may exhibit some degree of error. As demonstrated in Section 7.4, we consider a mean variation of up to 6.9% to be acceptable.
