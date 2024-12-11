# Rubick Testbed Experiments

## Contents
- `benckmark` and `sched/` contains code for testbed experiments and is adapted from [Pollux](https://github.com/petuum/adaptdl/tree/osdi21-artifact).
    - `benchmark/models/` contains the implementations of each evaluated model described in Table 2.
    - `benchmark/workloads/` contains the manually-tuned job trace used to evaluate Rubick, Sia and Synergy.
    - `benchmark/run_workload.py` submits jobs according to a workload trace.
    - `benchmark/run_monitor.py` monitors and logs the cluster state during each cluster scheduling experiment.
    - `rubick-sched/` contains testbed training code for testbed experiments.
    	- `rubick-sched/policy/rubick.py` containes the implementation of Rubick scheduling algorithm.
    	- `rubick-sched/policy/synergy.py` containes the implementation of Synergy scheduling algorithm provided by [Synergy](https://github.com/msr-fiddle/synergy).


## Reproduce testbed results

Note: Due to the execution scripts of testbed experiments are highly related to internal testbed platform, we only demonstrate the functionality and provide the reproduction steps on the hardware devices we use. Please adjust to your platform if you would like to execute the testbed experiment.

### Hardware

The testbed experiments require up to 8 VMs, each with 8 V100 GPUs, 64 CPU cores, 256 GB memory, and 20 Gbps network bandwidth. 
NAS is required for dataset and DL model checkpoint storage to speed up the I/O process. 
At least 100G NAS storage is needed on each node for the dataset and model checkpoints.
The provided scripts can be executed on Aliyun ecs.gn6v-c8g1.16xlarge VMs. Please adjust to your platform if you would like to execute the testbed experiment.

### Environment

1. Get the docker container.

Run `bash prepare_container.sh`.

Note that the docker size is 16.7GB, make sure that there is enough disk space. There should be a `/mnt` directory. NAS is required for this script.

2. Get the dataset.

### Reproduction Steps
#### Getting Started 

**Getting the scheduler images:**
```
cd sched
docker build -t alibaba/scheduler:rubick .
```

**Deploy System**
```
helm install rubick  helm/shed_deploy
```

**Undeploy System**
```
helm delete rubick
```