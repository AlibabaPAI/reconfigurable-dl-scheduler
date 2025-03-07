import collections
import copy
import heapq

res_type = ["gpu", "cpu", "mem"]
empty_res = {"gpu": 0, "cpu": 0, "mem": 0}


class MorphlingPolicy(object):
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
        total_res = {}
        free_res = {}

        # Add new jobs to pending.
        for key, job in job_infos.items():
            if key not in self._status:
                self._status[key] = "PENDING"
            if self._status[key] == "PENDING":
                self._queue.append(key)
            if self._status[key] == "RUNNING" and key not in self._queue:
                self._queue.append(key)
        for res in res_type:
            execution_plan[res] = {
                k: v for k, v in prev_execution_plan[res].items() if k in job_infos
            }
            total_res[res] = {
                idx: int(node.resources[res]) for idx, node in node_infos.items()
            }
            free_res[res] = collections.Counter(total_res[res]) - collections.Counter(
                sum(execution_plan[res].values(), [])
            )
        job_alloc = self.get_job_alloc(execution_plan)

        execution_plan["exec_plan"] = {
            k: v for k, v in prev_execution_plan["exec_plan"].items() if k in job_infos
        }
        job_throughput = self.job_res_slope(self._queue, job_infos, job_alloc)
        sorted_job = sorted(
            job_throughput, key=lambda k: job_throughput[k], reverse=False
        )
        is_sla = []
        notSLA = []

        for job in sorted_job:
            if job_infos[job].is_sla:
                is_sla.append(job)
            else:
                notSLA.append(job)
        job_list = is_sla + notSLA

        for job in job_list:
            if (
                job_infos[job].model_name in ["llama7", "llama30"]
                and job in execution_plan["gpu"]
                and len(execution_plan["gpu"][job]) >= 32
                and (job_infos[job].restarts < 7 or len(is_sla) > 12)
            ):
                execution_plan["gpu"][job] = []
                execution_plan["exec_plan"][job] = ()

        for job in job_list:
            if job not in job_alloc:
                job_alloc[job] = {}
                job_alloc[job]["gpu"] = 0
            if job not in execution_plan["gpu"]:
                execution_plan["gpu"][job] = []
                execution_plan["exec_plan"][job] = ()
            for res in res_type:
                free_res[res] = collections.Counter(
                    total_res[res]
                ) - collections.Counter(sum(execution_plan[res].values(), []))
            execution_plan = self.schedule_job(
                job_infos, job, free_res, node_infos, execution_plan, job_alloc
            )

        free_res[res] = collections.Counter(total_res[res]) - collections.Counter(
            sum(execution_plan[res].values(), [])
        )
        for node_idx in node_infos:
            targeted_nodes, node_total_res = self.get_targeted_nodes(
                execution_plan, node_idx, node_infos
            )
            execution_plan = self.scale_up(
                execution_plan, targeted_nodes, node_idx, job_infos, job_alloc, free_res
            )
        for job in sorted_job:
            if job not in job_alloc:
                job_alloc[job] = {}
                job_alloc[job]["gpu"] = 0
            if job not in execution_plan["gpu"]:
                execution_plan["gpu"][job] = []
                execution_plan["exec_plan"][job] = ()
            for res in res_type:
                free_res[res] = collections.Counter(
                    total_res[res]
                ) - collections.Counter(sum(execution_plan[res].values(), []))
            if job_infos[job].model_name in ["llama7", "llama30", "t5_3B"]:
                is_alloc, execution_plan_tmp = self.allocate_for_3D_parallelism(
                    copy.deepcopy(free_res),
                    copy.deepcopy(execution_plan),
                    job,
                    job_infos,
                    True,
                    total_res,
                )
                if is_alloc:
                    self._status[job] = "RUNNING"
                    execution_plan = execution_plan_tmp
                    job_alloc = self.get_job_alloc(execution_plan)

        return execution_plan, None

    def schedule_job(
        self, job_infos, job, free_res, node_infos, execution_plan, job_alloc
    ):
        if job_infos[job].model_name not in ["llama7", "llama30", "t5_3B"]:
            free_node_list = {i: 0 for i in range(0, 8)}
            for k in free_res["gpu"]:
                free_node_list[k] = free_res["gpu"][k]
            free_node_list_sorted = sorted(
                free_node_list, key=lambda k: free_node_list[k], reverse=True
            )

            for idx in free_node_list_sorted:
                if (
                    len(execution_plan["gpu"][job]) > 0
                    and idx not in execution_plan["gpu"][job]
                ):
                    continue
                targeted_nodes, _ = self.get_targeted_nodes(
                    execution_plan, idx, node_infos
                )
                is_alloc, execution_plan_tmp = self.alloc_per_node(
                    copy.deepcopy(free_res),
                    idx,
                    job_infos,
                    job_alloc,
                    targeted_nodes,
                    job,
                    execution_plan,
                )
                if is_alloc:
                    execution_plan = execution_plan_tmp
                    job_alloc = self.get_job_alloc(execution_plan)
                    self._status[job] = "RUNNING"
                    break
        else:
            # allocate model for 3D parallelism
            is_alloc, execution_plan_tmp = self.allocate_for_3D_parallelism(
                copy.deepcopy(free_res),
                copy.deepcopy(execution_plan),
                job,
                job_infos,
                False,
            )
            if is_alloc:
                self._status[job] = "RUNNING"
                execution_plan = execution_plan_tmp
                job_alloc = self.get_job_alloc(execution_plan)
        return execution_plan

    def scale_up(
        self, execution_plan, targeted_nodes, node_idx, job_infos, job_alloc, free_res
    ):
        if targeted_nodes == {}:
            return execution_plan
        nodeRes = free_res["gpu"][node_idx]
        alter_job = {}
        should_end = False
        while nodeRes > 0 and not should_end:
            job_throughput = self.job_res_slope(
                list(targeted_nodes.keys()), job_infos, job_alloc, nodeRes
            )
            sorted_job = sorted(
                job_throughput, key=lambda k: job_throughput[k], reverse=True
            )
            for i in range(0, len(sorted_job)):
                targeted_job = sorted_job[i]
                if i == len(sorted_job) - 1:
                    should_end = True
                if "llama" in targeted_job:
                    continue
                originRes = job_alloc[targeted_job]["gpu"]
                alter_job[targeted_job] = job_infos[targeted_job].res_matrix[
                    min(job_alloc[targeted_job]["gpu"] + nodeRes, 8)
                ][1][0][2]["gpu"]
                nodeRes -= alter_job[targeted_job] - originRes
                if alter_job[targeted_job] - originRes == 0:
                    should_end = True
                break
        if alter_job != {}:
            execution_plan = self.adapt_execution_plan(
                alter_job, copy.deepcopy(execution_plan), node_idx, job_infos
            )

        return execution_plan

    def alloc_per_node(
        self,
        free_res,
        node_idx,
        job_infos,
        job_alloc,
        targeted_nodes,
        job,
        execution_plan,
    ):
        freeRes = free_res["gpu"][node_idx]
        alter_job = {}
        if "gpu" in job_alloc[job]:
            origin_gpu = execution_plan["gpu"][job].count(node_idx)
            alter_job[job] = freeRes + origin_gpu
        else:
            origin_gpu = 0
            alter_job[job] = freeRes
        origin = alter_job[job]
        job_slope = self.job_res_slope(
            list(targeted_nodes.keys()), job_infos, job_alloc, -1
        )
        if job_slope != {}:
            # GetLowestSlopeOverMinJob
            sorted_job = sorted(job_slope, key=lambda k: job_slope[k])
            for i in range(0, len(sorted_job)):
                targeted_job = sorted_job[i]
                if "llama" or "3B" in targeted_job or targeted_job == job:
                    continue
                if alter_job[job] != origin and not job_infos[job].is_sla:
                    break
                elif job_infos[job].is_sla and (
                    job_infos[job].res_matrix[alter_job[job]][1][0][0] * 1.5
                    >= job_infos[job].sla_perf
                    or alter_job[job] >= job_infos[job].gpu_demand
                ):
                    break
                r = -1
                for r in [-1, -2, -4, -8]:
                    cur_slope = self.job_res_slope([job], job_infos, job_alloc, -r)
                    if (
                        job_slope[targeted_job] <= cur_slope[job]
                        and job_infos[targeted_job].restarts < 5
                        and job_infos[job].restarts < 5
                    ) or (
                        job_infos[job].is_sla
                        and job_infos[job].res_matrix[alter_job[job]][1][0][0]
                        < job_infos[job].sla_perf
                    ):
                        if job_alloc[targeted_job]["gpu"] + r == 0:
                            alter_job[targeted_job] = 0
                        elif job_alloc[targeted_job]["gpu"] + r < 0:
                            break
                        else:
                            alter_job[targeted_job] = job_infos[
                                targeted_job
                            ].res_matrix[job_alloc[targeted_job]["gpu"] + r][1][0][2][
                                "gpu"
                            ]
                        alter_job[job] = job_infos[job].res_matrix[min(origin - r, 8)][
                            1
                        ][0][2]["gpu"]
                        if job_infos[job].is_sla and (
                            job_infos[job].res_matrix[alter_job[job]][1][0][0] * 1.5
                            >= job_infos[job].sla_perf
                            or alter_job[job] >= job_infos[job].gpu_demand
                        ):
                            if not job_infos[targeted_job].is_sla:
                                break
                            if (
                                job_infos[targeted_job].res_matrix[
                                    alter_job[targeted_job]
                                ][1][0][0]
                                * 1.5
                                >= job_infos[targeted_job].sla_perf
                                or alter_job[targeted_job]
                                >= job_infos[targeted_job].gpu_demand
                            ):
                                break
                            else:
                                alter_job[job] = origin
                                del alter_job[targeted_job]
                        elif not job_infos[job].is_sla and alter_job[job] != origin:
                            if not job_infos[targeted_job].is_sla:
                                continue
                            elif (
                                job_infos[targeted_job].res_matrix[
                                    alter_job[targeted_job]
                                ][1][0][0]
                                * 1.5
                                >= job_infos[targeted_job].sla_perf
                                or alter_job[targeted_job]
                                >= job_infos[targeted_job].gpu_demand
                            ):
                                break
                            alter_job[job] = origin

        if (
            job_infos[job].is_sla
            and 1.5 * job_infos[job].res_matrix[alter_job[job]][1][0][0]
            >= job_infos[job].sla_perf
        ) or (not job_infos[job].is_sla and alter_job[job] != origin_gpu):
            alter_job[job] = job_infos[job].res_matrix[alter_job[job]][1][0][2]["gpu"]
            execution_plan = self.adapt_execution_plan(
                alter_job, copy.deepcopy(execution_plan), node_idx, job_infos
            )
            return True, execution_plan
        return False, execution_plan

    def adapt_execution_plan(self, alter_job, execution_plan, node_idx, job_infos):
        for job in alter_job:
            if alter_job[job] == 0:
                execution_plan["gpu"][job] = []
                execution_plan["exec_plan"][job] = ()
                self._status[job] = "PENDING"
                if job not in self._queue:
                    self._queue.append(job)
                continue
            execution_plan["gpu"][job] = [
                value for value in execution_plan["gpu"][job] if value != node_idx
            ]
            execution_plan["gpu"][job].extend([node_idx] * int(alter_job[job]))
            execution_plan["exec_plan"][job] = job_infos[job].res_matrix[
                alter_job[job]
            ][1][0][1]
        return execution_plan

    def job_res_slope(self, queue, job_infos, job_alloc, delta=1):
        job_slope = {}
        for job in queue:
            if job_infos[job].age > 200000:
                job_slope[job] = 1e9
            else:
                if job not in job_alloc:
                    job_slope[job] = abs(
                        abs(0 - job_infos[job].res_matrix[delta][1][0][0])
                        / job_infos[job].origin_tpt
                        / delta
                    )
                else:
                    job_slope[job] = abs(
                        abs(
                            job_infos[job].res_matrix[job_alloc[job]["gpu"]][1][0][0]
                            - job_infos[job].res_matrix[
                                min(job_alloc[job]["gpu"] + delta, 8)
                            ][1][0][0]
                        )
                        / job_infos[job].origin_tpt
                        / delta
                    )
        return job_slope

    def get_job_alloc(self, execution_plan):
        job_alloc = {}
        for res in res_type:
            for k, v in execution_plan[res].items():
                if k not in job_alloc:
                    job_alloc[k] = {}
                job_alloc[k][res] = len(v) / (
                    max(1, len(set(execution_plan[res][k])))
                    if res == "bandwidth"
                    else 1
                )
        return job_alloc

    def get_targeted_nodes(self, execution_plan, node_idx, node_infos):
        targeted_nodes = {}
        node_total_res = {}
        for res in res_type:
            for k, v in execution_plan[res].items():
                node_total_res[res] = node_infos[node_idx].resources[res]
                if node_idx in v:
                    if k not in targeted_nodes:
                        targeted_nodes[k] = copy.deepcopy(empty_res)
                    targeted_nodes[k][res] = v.count(node_idx)
        return targeted_nodes, node_total_res

    def allocate_for_3D_parallelism(
        self,
        free_res,
        execution_plan,
        job,
        job_infos,
        is_scale_up=False,
        total_res=None,
    ):
        # allocate model for 3D parallelism
        gpu_to_placements = job_infos[job].gpu_to_placements
        if is_scale_up:
            gpu_to_placements = dict(
                sorted(
                    gpu_to_placements.items(),
                    key=lambda item: item[0],
                    reverse=True,
                )
            )
        placements_to_plan = job_infos[job].placements_to_plan
        for key in gpu_to_placements:
            if is_scale_up:
                if len(execution_plan["gpu"][job]) > key:
                    continue
                execution_plan["gpu"][job] = []
                execution_plan["exec_plan"][job] = ()
                for res in res_type:
                    free_res[res] = collections.Counter(
                        total_res[res]
                    ) - collections.Counter(sum(execution_plan[res].values(), []))

            for placements in gpu_to_placements[key]:
                placement = [int(char) for char in str(placements)]
                placement = sorted(placement, reverse=True)
                norm_demand_vector = [1, 1, 1]
                if key in [1, 2, 8]:
                    norm_demand_vector[1] = 4
                is_alloc, res_alloc_tmp = self.allocate_3D(
                    norm_demand_vector, free_res, key, placement
                )
                if is_alloc:
                    for i in range(0, len(res_type)):
                        execution_plan[res_type[i]][job] = []
                    if (
                        sum(res_alloc_tmp[node_idx] for node_idx in res_alloc_tmp)
                        == key
                    ):
                        for i in range(0, len(res_type)):
                            for node_idx, num in res_alloc_tmp.items():
                                execution_plan[res_type[i]][job].extend(
                                    [node_idx] * num * int(norm_demand_vector[i])
                                )
                        if placements_to_plan[placements] == "zero-offload":
                            execution_plan["exec_plan"][job] = (
                                "DDP",
                                "ga",
                                "zero-offload",
                            )
                        else:
                            execution_plan["exec_plan"][job] = (
                                placements_to_plan[placements],
                                placements_to_plan[placements],
                                placements,
                            )
                        return True, execution_plan
        return False, execution_plan

    def allocate_3D(self, norm_demand_vector, free_res, job_gpu_deficit, placement):
        pq = [(num, node_idx) for node_idx, num in free_res["gpu"].items()]
        is_alloc, res_alloc = self._fit_gpu_3D(
            copy.deepcopy(placement),
            job_gpu_deficit,
            copy.deepcopy(pq),
            copy.deepcopy(free_res),
            norm_demand_vector,
        )
        if not is_alloc:
            return False, None
        else:
            return True, res_alloc

    def _fit_gpu_3D(
        self, placement, num_gpus, pq, free_res, job_demand_vector_gpu_norm
    ):
        res_addition = {}
        while (
            sum(res_addition[node_idx] for node_idx in res_addition) < num_gpus
            and len(pq) > 0
        ):
            _, node_idx = heapq.heappop(pq)
            if node_idx not in res_addition:
                res_addition[node_idx] = 0
            free_vector = [free_res[res][node_idx] for res in res_type]
            for p_idx in range(0, len(placement)):
                if placement[p_idx] == 0:
                    continue
                if placement[p_idx] <= free_vector[0]:
                    res_addition[node_idx] += placement[p_idx]
                    for j in range(0, placement[p_idx]):
                        for i in range(0, len(res_type)):
                            free_res[res_type[i]][
                                node_idx
                            ] -= job_demand_vector_gpu_norm[i]
                    placement[p_idx] = 0
                    break
        if sum(res_addition[node_idx] for node_idx in res_addition) >= num_gpus:
            return True, res_addition
        return False, None
