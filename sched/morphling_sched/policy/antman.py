import collections
from model import Model
import copy
import math
import heapq

rsc_type = ["nvidia.com/gpu"]
empty_rsc = {"nvidia.com/gpu": 0}


class antmanPolicy(object):
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        # Utilization thresholds for cluster autoscaling.
        self._status = {}
        self._queue = []

    def optimize(self, job_infos, node_infos, prev_execution_plan):
        self._status = {
            key: val for key, val in self._status.items() if key in job_infos
        }
        self._queue = [key for key in self._queue if key in job_infos]
        execution_plan = {}
        job_alloc = {}
        total_rsc = {}
        free_rsc = {}
        node_total_rsc = {}
        for key, job in job_infos.items():
            if key not in self._status:
                self._status[key] = "PENDING"
                self._queue.append(key)
            if self._status[key] == "RUNNING" and key not in self._queue:
                self._queue.append(key)
        for rsc in rsc_type:
            execution_plan[rsc] = {
                k: v for k, v in prev_execution_plan[rsc].items() if k in job_infos
            }
            total_rsc[rsc] = {
                idx: int(node.resources[rsc]) for idx, node in node_infos.items()
            }
            free_rsc[rsc] = collections.Counter(total_rsc[rsc]) - collections.Counter(
                sum(execution_plan[rsc].values(), [])
            )
        job_alloc = self.get_job_alloc(execution_plan)

        isSLA = []
        notSLA = []
        for job in self._queue:
            if job_infos[job].isSLA:
                isSLA.append(job)
            else:
                notSLA.append(job)
                if (
                    job_infos[job].model_name in ["llama7", "llama30"]
                    and job in execution_plan["nvidia.com/gpu"]
                ):
                    execution_plan["nvidia.com/gpu"][job] = []
        job_list = isSLA + notSLA
        print("job_list", job_list)
        for job in job_list:
            is_allocated = False
            if job not in job_alloc:
                job_alloc[job] = {}
                job_alloc[job]["nvidia.com/gpu"] = 0
            if job not in execution_plan["nvidia.com/gpu"]:
                execution_plan["nvidia.com/gpu"][job] = []
            for rsc in rsc_type:
                free_rsc[rsc] = collections.Counter(
                    total_rsc[rsc]
                ) - collections.Counter(sum(execution_plan[rsc].values(), []))

            if (
                job_infos[job].model_name not in ["llama7", "llama30"]
                and not is_allocated
            ):
                free_node_list = {i: 0 for i in range(0, 8)}
                for k in free_rsc["nvidia.com/gpu"]:
                    free_node_list[k] = free_rsc["nvidia.com/gpu"][k]
                free_node_list_sorted = sorted(
                    free_node_list, key=lambda k: free_node_list[k], reverse=True
                )
                for idx in free_node_list_sorted:
                    if (
                        len(execution_plan["nvidia.com/gpu"][job]) > 0
                        and idx not in execution_plan["nvidia.com/gpu"][job]
                    ):
                        continue
                    targeted_nodes, _ = self.get_targeted_nodes(
                        execution_plan, idx, node_infos
                    )
                    is_alloc, execution_plan_tmp = self.schedule_on_node(
                        copy.deepcopy(free_rsc),
                        idx,
                        job_infos,
                        job_alloc,
                        targeted_nodes,
                        job,
                        execution_plan,
                    )
                    if is_alloc:
                        is_allocated = True
                        execution_plan = execution_plan_tmp
                        job_alloc = self.get_job_alloc(execution_plan)
                        self._status[job] = "RUNNING"
                        break
            elif (
                job_infos[job].model_name in ["llama7", "llama30"] and not is_allocated
            ):
                # allocate for llama 3D parallelism
                if job_infos[job].model_name == "llama7":
                    is_alloc, rsc_alloc_tmp, execution_plan_tmp = (
                        self.allocate_for_llama(
                            copy.deepcopy(free_rsc),
                            copy.deepcopy(execution_plan),
                            job,
                            job_infos,
                            False,
                        )
                    )
                    if is_alloc:
                        self._status[job] = "RUNNING"
                        is_allocated = True
                        execution_plan = execution_plan_tmp
                        job_alloc = self.get_job_alloc(execution_plan)
                elif job_infos[job].model_name == "llama30":
                    is_alloc, rsc_alloc_tmp, execution_plan_tmp = (
                        self.allocate_for_llama(
                            copy.deepcopy(free_rsc),
                            copy.deepcopy(execution_plan),
                            job,
                            job_infos,
                            False,
                        )
                    )
                    if is_alloc:
                        self._status[job] = "RUNNING"
                        is_allocated = True
                        execution_plan = execution_plan_tmp
                        job_alloc = self.get_job_alloc(execution_plan)

        free_rsc[rsc] = collections.Counter(total_rsc[rsc]) - collections.Counter(
            sum(execution_plan[rsc].values(), [])
        )
        for node_idx in node_infos:
            targeted_nodes, node_total_rsc = self.get_targeted_nodes(
                execution_plan, node_idx, node_infos
            )
            execution_plan = self.expand(
                execution_plan, targeted_nodes, node_idx, job_infos, job_alloc, free_rsc
            )

        return execution_plan, 0

    def expand(
        self, execution_plan, targeted_nodes, node_idx, job_infos, job_alloc, free_rsc
    ):
        if targeted_nodes == {}:
            return execution_plan
        nodeRes = free_rsc["nvidia.com/gpu"][node_idx]
        alter_job = {}
        should_end = False
        for targeted_job in targeted_nodes.keys():
            if "llama" in targeted_job[1]:
                continue
            originRes = job_alloc[targeted_job]["nvidia.com/gpu"]
            print(originRes, job_alloc[targeted_job]["nvidia.com/gpu"], nodeRes)
            alter_job[targeted_job] = self.adjust_gpu_num(
                job_infos,
                targeted_job,
                min(job_alloc[targeted_job]["nvidia.com/gpu"] + nodeRes, 8),
            )
            nodeRes -= alter_job[targeted_job] - originRes
            break
        if alter_job != {}:
            execution_plan = self.adapt_execution_plan(
                alter_job, copy.deepcopy(execution_plan), node_idx, job_infos
            )

        return execution_plan

    def adjust_gpu_num(self, job_infos, job, num_gpus):
        if job_infos[job].model_name == "gpt2":
            if job_infos[job].strategy == "151":
                if num_gpus >= 5:
                    return 5
                else:
                    return 0
            elif job_infos[job].strategy == "ga":
                if num_gpus <= 1:
                    return 0
        if num_gpus < 1:
            return 0
        elif num_gpus < 2:
            return 1
        elif num_gpus <= 3:
            return 2
        elif num_gpus <= 7:
            return 4
        elif num_gpus >= 8:
            return 8

    def schedule_on_node(
        self,
        free_rsc,
        node_idx,
        job_infos,
        job_alloc,
        targeted_nodes,
        job,
        execution_plan,
    ):
        freeRes = free_rsc["nvidia.com/gpu"][node_idx]
        alter_job = {}
        if "nvidia.com/gpu" in job_alloc[job]:
            origin_gpu = execution_plan["nvidia.com/gpu"][job].count(node_idx)
            alter_job[job] = freeRes + origin_gpu
        else:
            origin_gpu = 0
            alter_job[job] = freeRes
        origin = alter_job[job]
        if (
            job_infos[job].isSLA and origin >= job_infos[job].job_gpu_demand
        ) or not job_infos[job].isSLA:
            alter_job[job] = self.adjust_gpu_num(job_infos, job, alter_job[job])
            execution_plan = self.adapt_execution_plan(
                alter_job, copy.deepcopy(execution_plan), node_idx, job_infos
            )
            return True, execution_plan

        for targeted_job in targeted_nodes.keys():
            if "llama" in targeted_job[1] or targeted_job == job:
                continue
            elif job_infos[job].isSLA and (
                alter_job[job] >= job_infos[job].job_gpu_demand
            ):
                break
            for r in [-1, -2, -4, -8]:
                if (job_infos[targeted_job].restarts < 5) and job_infos[job].isSLA:
                    if job_alloc[targeted_job]["nvidia.com/gpu"] + r == 0:
                        alter_job[targeted_job] = 0
                    elif job_alloc[targeted_job]["nvidia.com/gpu"] + r < 0:
                        break
                    else:
                        alter_job[targeted_job] = self.adjust_gpu_num(
                            job_infos,
                            job,
                            job_alloc[targeted_job]["nvidia.com/gpu"] + r,
                        )
                    alter_job[job] = self.adjust_gpu_num(
                        job_infos, job, min(origin - r, 8)
                    )
                    if (
                        job_infos[job].isSLA
                        and alter_job[job] >= job_infos[job].job_gpu_demand
                    ):
                        if not job_infos[targeted_job].isSLA:
                            break
                        if (
                            alter_job[targeted_job]
                            >= job_infos[targeted_job].job_gpu_demand
                        ):
                            break
                        else:
                            alter_job[job] = origin
                            del alter_job[targeted_job]

        if job_infos[job].isSLA and (alter_job[job] >= job_infos[job].job_gpu_demand):
            alter_job[job] = self.adjust_gpu_num(job_infos, job, alter_job[job])
            execution_plan = self.adapt_execution_plan(
                alter_job, copy.deepcopy(execution_plan), node_idx, job_infos
            )
            return True, execution_plan
        return False, execution_plan

    def adapt_execution_plan(self, alter_job, execution_plan, node_idx, job_infos):
        for job in alter_job:
            if alter_job[job] == 0:
                execution_plan["nvidia.com/gpu"][job] = []
                self._status[job] = "PENDING"
                if job not in self._queue:
                    self._queue.append(job)
                continue
            execution_plan["nvidia.com/gpu"][job] = [
                value
                for value in execution_plan["nvidia.com/gpu"][job]
                if value != node_idx
            ]
            execution_plan["nvidia.com/gpu"][job].extend(
                [node_idx] * int(alter_job[job])
            )
        return execution_plan

    def get_job_alloc(self, execution_plan):
        job_alloc = {}
        for rsc in rsc_type:
            for k, v in execution_plan[rsc].items():
                if k not in job_alloc:
                    job_alloc[k] = {}
                job_alloc[k][rsc] = len(v) / (
                    max(1, len(set(execution_plan[rsc][k])))
                    if rsc == "bandwidth"
                    else 1
                )
        return job_alloc

    def get_targeted_nodes(self, execution_plan, node_idx, node_infos):
        targeted_nodes = {}
        node_total_rsc = {}
        for rsc in rsc_type:
            for k, v in execution_plan[rsc].items():
                node_total_rsc[rsc] = node_infos[node_idx].resources[rsc]
                if node_idx in v:
                    if k not in targeted_nodes:
                        targeted_nodes[k] = copy.deepcopy(empty_rsc)
                    targeted_nodes[k][rsc] = v.count(node_idx)
        return targeted_nodes, node_total_rsc

    def allocate_for_llama(
        self, free_rsc, execution_plan, job, job_infos, expand=False, total_rsc=None
    ):
        if job_infos[job].strategy == "zero-offload":
            norm_demand_vector = [1, 4, 1]
            is_alloc, rsc_alloc_tmp = self.allocate_3d(
                [1, 4, 1],
                free_rsc,
                job_infos[job].job_gpu_demand,
                [job_infos[job].job_gpu_demand],
            )
            if is_alloc:
                for i in range(0, len(rsc_type)):
                    execution_plan[rsc_type[i]][job] = []
                if (
                    sum(rsc_alloc_tmp[node_idx] for node_idx in rsc_alloc_tmp)
                    == job_infos[job].job_gpu_demand
                ):
                    for i in range(0, len(rsc_type)):
                        for node_idx, num in rsc_alloc_tmp.items():
                            execution_plan[rsc_type[i]][job].extend(
                                [node_idx] * num * int(norm_demand_vector[i])
                            )

                    return True, rsc_alloc_tmp, execution_plan
            return False, None, execution_plan
        for placements in job_infos[job].avail_placement:

            key = job_infos[job].job_gpu_demand
            placement = [int(char) for char in str(placements)]
            placement = sorted(placement, reverse=True)
            norm_demand_vector = [1, 1, 1]
            is_alloc, rsc_alloc_tmp = self.allocate_3d(
                norm_demand_vector, free_rsc, key, placement
            )
            if is_alloc:
                for i in range(0, len(rsc_type)):
                    execution_plan[rsc_type[i]][job] = []
                if sum(rsc_alloc_tmp[node_idx] for node_idx in rsc_alloc_tmp) == key:
                    for i in range(0, len(rsc_type)):
                        for node_idx, num in rsc_alloc_tmp.items():
                            execution_plan[rsc_type[i]][job].extend(
                                [node_idx] * num * int(norm_demand_vector[i])
                            )

                    return True, rsc_alloc_tmp, execution_plan
        return False, None, execution_plan

    def allocate_3d(self, norm_demand_vector, free_rsc, job_gpu_deficit, placement):
        pq = [(num, node_idx) for node_idx, num in free_rsc["nvidia.com/gpu"].items()]
        is_alloc, rsc_alloc = self._fit_gpu_3d(
            copy.deepcopy(placement),
            job_gpu_deficit,
            copy.deepcopy(pq),
            copy.deepcopy(free_rsc),
            norm_demand_vector,
        )
        if not is_alloc:
            return False, None
        else:
            return True, rsc_alloc

    def _fit_gpu_3d(
        self, placement, num_gpus, pq, free_rsc, job_demand_vector_gpu_norm
    ):
        rsc_addition = {}
        while (
            sum(rsc_addition[node_idx] for node_idx in rsc_addition) < num_gpus
            and len(pq) > 0
        ):
            _, node_idx = heapq.heappop(pq)
            if node_idx not in rsc_addition:
                rsc_addition[node_idx] = 0
            free_vector = [free_rsc[rsc][node_idx] for rsc in rsc_type]
            for p_idx in range(0, len(placement)):
                if placement[p_idx] == 0:
                    continue
                if placement[p_idx] <= free_vector[0]:
                    rsc_addition[node_idx] += placement[p_idx]
                    for j in range(0, placement[p_idx]):
                        for i in range(0, len(rsc_type)):
                            free_rsc[rsc_type[i]][
                                node_idx
                            ] -= job_demand_vector_gpu_norm[i]
                    placement[p_idx] = 0
                    break
        if sum(rsc_addition[node_idx] for node_idx in rsc_addition) >= num_gpus:
            return True, rsc_addition
        return False, None
