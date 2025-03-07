# Artifact for Morphling (MLSys'25)

## Overview
This is the artifact for the MLSys'25 paper "Morphling: Exploiting Job Reconfigurability for Deep Learning Cluster Scheduling".

You will be able to verify the functionality of the Morphling and reproduce the results in the paper by following the instructions provided below in order.

Besides, we provide the code organization of Morphling at the end of this document. You can check the functionality of Morphling by reviewing the codes.


> For AE Reviewers, 
>
> Morphling can be deployed in real GPU clusters. In our paper, we validated Morphling’s advantages using a GPU cluster equipped with 64 NVIDIA A800 GPUs. However, reproducing the Table 4 in Section 7.3 requires access to the internal 64-GPU cluster in our organization, and the entire experiemnts are both costly and time-consuming (the sum of `Makespan (h)` in Table 4 means that it will take at least 9 days to complete).
>
> To facilitate the rapid verification of Morphling’s core functionalities, we reproduce the perfermance model validations and micro-benchmarks in the paper with the help of throughput values we collect in advance. And we also provide a cluster simulator that emulates Morphling’s scheduling in the GPU cluster. More details can be found in the `simulator/README.md`. It is important to highlight that all the code in the artifact involving core components of Morphling (including scheduling algorithms and performance models) is consistent with the that used in real GPU clusters.
>
> Additionally, we are pleased to offer code and its orgainization to help Morphling setup in a real GPU cluster. These details can be found under `sched/`.

## Artifact Setup

### For AE Reviewers (Using Public Docker Containers)

We have already setup the Docker containers and pushed to the public repository. The reviewers just need to pull the container and start the container:

```
docker pull zzxy180318/morphling-artifact:mlsys25ae
docker run -tid --name morphling-artifact morphling:mlsys25ae
docker exec -it morphling:mlsys25ae /bin/bash
```

That's it! You can then jump to the instructions for running the experiments.


### Prepare Environment By Yourself

You can also setup the containers by yourself by using Dockerfile:

```
docker build -t morphling:mlsys25ae .
docker run -tid --name morphling-artifact morphling:mlsys25ae
docker exec -it morphling-artifact /bin/bash
```

## Getting Started Instructions (a naive scheduling example)

To verify that the environment has been successfully built and to check the basic functionality of the artifact, we provided a naive example that can be finished in less than 1 miniutes.

```
cd artifact/0_getting_started

# Submit 10 workloads, schedule them with Morphling and wait them for completion.
./morphling_exp simulator/workloads/naive.csv morphling 60 8 8 96 1600 400 100 0 'naive'
```

During the execution of the script, Morphling continuously prints the active jobs, GPU utilization, completed jobs and average job completion time at each simulator interval. You should be able to see logs like:

```
...
---------------- SIMULATOR TIME: xxxx ----------------
Active jobs:
    llama30-8:  [restarts x]    [placement (x,x,x,x)]
GPU utilization: xx
Completed jobs:
{'xx-0': xxxx, 'xx-1': xxxx, ...}
Average JCT (s): xxxxx
Makespan (s): xxxx
...

```

The last `Average JCT` and `Makespan` is the average job completion time and the makespan of the trace, respectively.

If the above command is executed successfully without getting stucked for a long time, it means that we have built the environment correctly and can go to the next detailed instructions section.

## Detailed Instuctions (to validate the functions and reproduce the evaluation results)

**The estimated time of running all experiments below for once is about 2 hours.**

**You can use the \*.ipynb files for the experiments we provided. Alternatively, you can also directly view the  \*.md files to get execution of the *.ipynb files and obtain the experimental results in Section 7.1 and Section 7.2.**



### 7.1 Performance Model Validation (Table 2)


* Run script to generate a new Notebook file that includes the output.

```
cd ./artifact/71_performance_model
jupyter nbconvert --execute validation.ipynb --to notebook
```

* Alternatively, you can directly run all cells by clicking  `Cell > Run All` on the menu  in Jupyter Notebook or JupyterLab.


We can get all the results in Table 2 of the paper through the steps above.

First, validation.ipynb fits all the transformer models in Table 1 using limited profile data (in `./artifact/71_performance_model/model/profile`). These throughput values are collected from several sampled test runs in advance using different resource allocations and execution plan. It then uses the newly-generated fittable parameters to predict the performance of these models. It compares the prediction with the actual performance of the models, and generate the table for each model to show the results.

You can also run the validation experiments for each model individually to view the model parameter fitting results and performance prediction results. The file contains two functions, `fit_XX()` for parameter fitting and `validate()`  for performance prediction.
* Run script:
```
cd ./artifact/71_performance_model/model
# All the files formalized as validate_{model_name}.py can be executed in the following way 
python validate_bert.py
python validate_gpt.py
...
```


### 7.2 Micro-benchmarks: Adapting to changing resource limits (Figure 7)

One of the advantages of Morphling is its reconfiguarability to always choose the best execution plan under different resource limits. This experiment tests the reconfigurability of Morphling by continuously decreasing the limits of available resources.

* Run script to generate a new Notebook file that includes the output.

```
cd ./artifact/72_micro_benchmarks
jupyter nbconvert --execute Figure7.ipynb --to notebook
```

* Alternatively, you can directly run all cells by clicking  `Cell > Run All` in Jupyter Notebook or JupyterLab.

We can get Figure 7 of the paper through the steps above.


First, we construct the performance model of Morphling. The LLaMA-2-7B model configuration can be found in `./artifact/72_micro_benchmarks/llama7_config.py`. Then, we input different resource combinations (i.e. GPU placement) into the performance model, the model searches for the highest training throughput and get its corresponding execution plan. The detailed reconfiguration results will be printed during the execution.

Figure7.ipynb gets the training throughput of other execution plans through running them in advance, and the collected data is saved at `./simulator/traces/`. In the following experiments, this method is also used to directly obtain the performance of jobs under specific resource amounts and execution plans. Figure7.ipynb also generates a figure to discript the throughput variation of different execution plans when the resource limits vary.

### 7.2 Micro-benchmarks: Maximizing throughput across jobs (Figure 8)

Another function of Morphling is to maximize throughput considering jobs' resource sensitivity. The experiment tests it through submitting two jobs to a cluster of 4 A800 GPUs.

* Run script to generate a new Notebook file that includes the output.

```
cd ./artifact/72_micro_benchmarks
jupyter nbconvert --execute Figure8.ipynb --to notebook
```

* Alternatively, you can directly run all cells by clicking  `Cell > Run All` in Jupyter Notebook or JupyterLab.

We can get Figure 8 of the paper through the steps above.


### 7.2 Micro-benchmarks: Accuracy during reconfiguration (Figure 9)


Morphling keeps the global batch size unchanged during reconfiguration, ensuring training accuracy is not affected by design.

* Run script to generate a new Notebook file that includes the output.

```
cd ./artifact/72_micro_benchmarks
jupyter nbconvert --execute Figure9.ipynb --to notebook
```
* Alternatively, you can directly run all cells by clicking  `Cell > Run All` in Jupyter 
Notebook or JupyterLab.


### 7.3 Simualtion: Cluster Experiments

Our simulation experiments simulate the GPU cluster with 8 node, each containing 8 A800 GPUs, as decribed in section 7.4 of the paper.

We selected three experiments from Section 7.3 to validate the full capabilities of the Morphling (including the performance models and the scheduling algorithms). Specifically, we use the same `Base trace`, `Multi-tenant trace`, and `Best-plan trace` as in Table 4 to conduct end-to-end simulation cluster experiments. For detailed descriptions of these three traces, please refer to Section 7.3.


```
cd artifact/73_cluster_experiment

# Submit 406 workloads, schedule them with Morphling and wait them for completion.

# Base Trace
./morphling_exp simulator/workloads/workload-base.csv morphling 60 8 8 96 1600 400 100 0 'Base'

# BP (best-plan) Trace
./morphling_exp simulator/workloads/workload-bp.csv morphling 60 8 8 96 1600 400 100 0 'BP'

# MT (multi-tenant) Trace
./morphling_exp simulator/workloads/workload-mt.csv morphling 60 8 8 96 1600 400 100 0 'MT'
```

During the experiment, all the resource allocations (per-job resource amounts and placements) and job statuses (submission, queuing, running, completion) are saved in "artifact/73_cluster_experiment/{trace_name}_{date-time}.log". The final "Average JCT" and "Makespan" values in the logs represent the scheduling results for that trace, corresponding to the "Avg. JCT" and "Makespan" in Table 4. AE reviewers can feel free to view the logs and conclude that:

- The successful execution of the above experiments verifies that Morphling can effectively schedule a large number of jobs in the 64-GPU cluster. 
- All the three experiments validates Morphling's ability to reconfigure the plan together with resource scaling during the scheduling process, and the comparision with SOTA baselines in Table 4 demonstates that Morphling can maximize cluster throughput.
- The experiment using MT trace validates Morphling can provide performance guarantees to guaranteed jobs.

## Testbed Setup

The testbed experiments require 8 nodes, each with 8 NVIDIA A800 GPUs (80 GB), 96 vCPUs, 1,600 GB memory, 400 GB/s NVLink bandwidth, and 100 GB/s RDMA network bandwidth. The experiments in the paper are also highly related to internal testbed platform.

Please see `sched/README.md` for more details.


### Code Organization (to check functionality)
We list the code orgainzation of the Morphling project to help the AE reviewers quickly understand the roles of each part in the project.
```
- Morphling
  - artifact // Reproduce the evaluation results of the paper. Feel free to execute the code following the instructions above.
  - benchmark // Implement the transformer model in Table 1 and manage the workloads in real GPU cluster.
  - sched
    - models // The specification of the transformer model.
    - morphling-sched // Core functionalities of Morphling, including the scheduling algorithm and the performance model.
    - ... // Other files are mainly used to manage the workloads and resources in real GPU cluster, such as implementing the scheduling decision.
- simulator 
    - traces // The training throughput values collected in advance for each model in Table 1 with differnt resource amounts and execution plan.
    - .. // Other files are mainly used to manage the workloads and resources in simulation GPU cluster. Note that the simulator invokes the classes and functions in `sched/morphling-sched` to use Morphling.
  - ...
```