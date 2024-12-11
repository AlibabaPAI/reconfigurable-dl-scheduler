# Artifact for Rubick

We provide the artifact for the "Rubick: Exploiting Job Reconfigurability for Deep Learning Cluster Scheduling", including:

- The main implementation of Rubick.
- Testbed experiment scripts (Sec 7.2).
- Cluster simulation scripts (Sec 7.3), which get the main results of the paper.
- Figure plotting scripts.


## Testbed Experiments

Note: Due to the execution scripts of testbed experiments are highly related to internal testbed platform, we only demonstrate the functionality and provide the reproduction steps on the hardware devices we use. Please adjust to your platform if you would like to execute the testbed experiment.

The testbed experiments require 8 nodes, each with 8 NVIDIA A800 GPUs (80GB), 96 vCPUs, 1,600 GB memory, 400 GB/s NVLink bandwidth, and 100 GB/s RDMA network bandwidth.


### General Testbed Experiments
Please see `sched/README.md` for more details.

## Simulation Experiments

### General Simulation Experiments

Please see `simulator/README.md` for more details. 


## Plotting Figures
Please refer to `plot_figure/README.md`