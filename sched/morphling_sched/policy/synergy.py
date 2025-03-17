import collections
from enum import Enum
import heapq
import copy
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
# synergy:fifo+tune+not fair+random
# self.allocate_synergy_tune(available_gpus, num_gpus, job, time, alloc=random, fair=false)

rsc_type = ["nvidia.com/gpu", "cpu", "memory"]
lease_allocation_time = 3600


class AllocationStrategy(Enum):
    DEFAULT_ORDER = 1
    PLACEMENT_SENSITIVE = 2
    SYNERGY_RANDOM = 3
    SYNERGY_PLACEMENT = 4


class SynergyPolicy(object):
    def __init__(self):
        self._status = {}
        self._queue = []
        self.per_server_size_fair = None

    def optimize(self, job_infos, node_infos, prev_execution_plan):
        self._status = {
            key: val for key, val in self._status.items() if key in job_infos
        }
        self._queue = []
        free_rsc = {}
        total_rsc = {}
        execution_plan = {}
        for key, job in job_infos.items():
            if key not in self._status:
                self._status[key] = "PENDING"
            if self._status[key] == "PENDING":
                self._queue.append(key)
            if self._status[key] == "RUNNING" and (
                key not in prev_execution_plan["nvidia.com/gpu"]
                or len(prev_execution_plan["nvidia.com/gpu"][key]) == 0
            ):
                self._queue.append(key)

            # if self._status[key] == 'RUNNING' and key in self._queue and job.lease_time<lease_allocation_time:
            #    self._queue.pop(self._queue.index(key))
            if (
                self._status[key] == "RUNNING"
                and job.lease_time >= lease_allocation_time
                and key not in self._queue
            ):
                self._queue.append(key)

        for rsc in rsc_type:
            execution_plan[rsc] = {
                k: v
                for k, v in prev_execution_plan[rsc].items()
                if k in job_infos and k not in self._queue
            }
            total_rsc[rsc] = {
                idx: int(node.resources[rsc]) for idx, node in node_infos.items()
            }
            free_rsc[rsc] = collections.Counter(total_rsc[rsc]) - collections.Counter(
                sum(execution_plan[rsc].values(), [])
            )
        self.running_jobs = []
        self.running_job_ids = []
        skipped_jobs = []

        jobs_gpu = {}
        num_free_gpus = sum(
            free_rsc["nvidia.com/gpu"][node_idx]
            for node_idx in free_rsc["nvidia.com/gpu"]
        )
        for job in self._queue:
            job_gpu_deficit = job_infos[job].job_demand_vector[0]
            if job_gpu_deficit > 0 and num_free_gpus >= job_gpu_deficit:
                jobs_gpu[job] = job_gpu_deficit
                # num_free_gpus -= job_gpu_deficit

        # jobs_this_round=sorted(jobs_gpu.items(), key=lambda x: -x[1])
        for job_v in jobs_gpu:
            job = job_v
            for rsc in rsc_type:
                execution_plan[rsc][job] = []
            job_gpu_deficit = job_infos[job].job_demand_vector[0]
            if job_infos[job].strategy not in ["ga", "zero-offload", "zero-dp", "gc"]:
                avail_placement = job_infos[job].avail_placement
                print("avail_placement", avail_placement)
                for placement in avail_placement:
                    placement = [int(char) for char in str(placement)]
                    placement = sorted(placement, reverse=True)
                    if (
                        job_gpu_deficit > 0
                        and sum(
                            free_rsc["nvidia.com/gpu"][node_idx]
                            for node_idx in free_rsc["nvidia.com/gpu"]
                        )
                        >= job_gpu_deficit
                    ):
                        is_alloc, execution_plan_tmp, free_rsc_tmp = self.allocate_3d(
                            job_infos,
                            node_infos,
                            free_rsc,
                            copy.deepcopy(execution_plan),
                            job_gpu_deficit,
                            job,
                            placement,
                        )
                        if is_alloc:
                            self._status[job] = "RUNNING"
                            execution_plan = execution_plan_tmp
                            free_rsc = free_rsc_tmp
                            break
            else:
                if (
                    job_gpu_deficit > 0
                    and sum(
                        free_rsc["nvidia.com/gpu"][node_idx]
                        for node_idx in free_rsc["nvidia.com/gpu"]
                    )
                    >= job_gpu_deficit
                ):
                    is_alloc, execution_plan_tmp, free_rsc_tmp = self.allocate(
                        job_infos,
                        node_infos,
                        free_rsc,
                        copy.deepcopy(execution_plan),
                        job_gpu_deficit,
                        job,
                        fair=False,
                        tune=True,
                    )
                    if is_alloc:
                        self._status[job] = "RUNNING"
                        execution_plan = execution_plan_tmp
                        free_rsc = free_rsc_tmp
        return execution_plan, None

    def allocate_3d(
        self,
        job_infos,
        node_infos,
        free_rsc,
        execution_plan,
        job_gpu_deficit,
        job,
        placement,
    ):
        pq = [(num, node_idx) for node_idx, num in free_rsc["nvidia.com/gpu"].items()]
        job_demand_vector = job_infos[job].job_demand_vector
        job_demand_vector_gpu_norm = self.gpu_normalized_vector(job_demand_vector)
        is_alloc, rsc_addition, free_rsc_tmp = self._fit_gpu_3d(
            copy.deepcopy(placement),
            job_gpu_deficit,
            copy.deepcopy(pq),
            copy.deepcopy(free_rsc),
            job_demand_vector_gpu_norm,
        )
        if not is_alloc:
            return False, execution_plan, free_rsc
        else:
            free_rsc = copy.deepcopy(free_rsc_tmp)
            for i in range(0, len(rsc_type)):
                for node_idx, num in rsc_addition.items():
                    execution_plan[rsc_type[i]][job].extend(
                        [node_idx] * num * int(job_demand_vector_gpu_norm[i])
                    )
        return True, execution_plan, free_rsc

    def _fit_gpu_3d(
        self, placement, num_gpus, pq, free_rsc, job_demand_vector_gpu_norm
    ):
        rsc_addition = {}
        print("placements", placement)
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
            return True, rsc_addition, free_rsc
        return False, None, None

    def allocate(
        self,
        job_infos,
        node_infos,
        free_rsc,
        execution_plan,
        job_gpu_deficit,
        job,
        fair,
        tune,
    ):
        job_demand_vector = job_infos[job].job_demand_vector
        print("job_demand_vector", job, job_demand_vector)
        _call_allocate = self.allocate_synergy_random
        is_alloc, execution_plan, free_rsc = self._tune(
            job,
            job_infos,
            node_infos,
            job_demand_vector,
            execution_plan,
            job_gpu_deficit,
            False,
            True,
            False,
            _call_allocate,
            free_rsc,
            fair,
        )
        return is_alloc, execution_plan, free_rsc

    def _tune(
        self,
        job,
        job_infos,
        node_infos,
        demand_vec,
        execution_plan,
        job_gpu_deficit,
        peer_adjust,
        initial,
        final,
        _call_allocate,
        free_rsc,
        fair,
    ):
        success, execution_plan_tmp, free_rsc_tmp = _call_allocate(
            job_infos,
            node_infos,
            free_rsc,
            execution_plan,
            job_gpu_deficit,
            job,
            fair=fair,
            demand_vec=demand_vec,
        )
        print(job, success)
        if success:
            return True, execution_plan_tmp, free_rsc_tmp
        else:
            return False, execution_plan, free_rsc
        if final:
            return execution_plan_tmp, free_rsc
        # We could not allocate with the job's demands.
        # Switch job to fair-share

        can_adjust, new_demand_vec = self._make_fair_share(
            job_infos, job, demand_vec, node_infos
        )
        job_gpu_deficit = job_infos[job].job_demand_vector[0]
        print("why", can_adjust, new_demand_vec)

        if initial:
            if can_adjust:
                demand_vec = new_demand_vec
            return self._tune(
                job,
                job_infos,
                node_infos,
                demand_vec,
                execution_plan,
                job_gpu_deficit,
                False,
                False,
                False,
                _call_allocate,
                free_rsc,
                fair,
            )
        elif not can_adjust and not peer_adjust:
            # Current job's demands amade fair-share already
            # Peer has not been adjusted yet
            return self._tune(
                job,
                job_infos,
                node_infos,
                demand_vec,
                execution_plan,
                job_gpu_deficit,
                True,
                False,
                False,
                _call_allocate,
                free_rsc,
                fair,
            )
        elif peer_adjust and not final:
            # Get servers with underutilized GPU (randomly)
            nodes_map = self._get_underutilized_servers(job_gpu_deficit, free_rsc)
            for node_idx, gpus in nodes_map.items():
                free_vec = [free_rsc[rsc][node_idx] for rsc in rsc_type]
                ratio = gpus / job_gpu_deficit
                demand_vec_share = [res * ratio for res in demand_vec]
                jobs_to_realloc = self._reallocate_peer(
                    demand_vec_share, free_vec, node_idx, execution_plan
                )

                for j in jobs_to_realloc:
                    gpus_realloc = len(execution_plan["nvidia.com/gpu"][j])
                    peer_res_map = {}
                    for node_idx in set(execution_plan["nvidia.com/gpu"][j]):
                        peer_res_map[node_idx] = {
                            rsc: len(execution_plan[rsc][j]) for rsc in rsc_type
                        }
                    for rsc in rsc_type:
                        execution_plan[rsc][j] = []

                    # We deallocated it . So get demand vec
                    peer_demand_vec = job_infos[j].job_demand_vector
                    can_adjust, new_demand_vec_peer = self._make_fair_share(
                        job_infos, j, peer_demand_vec, node_infos
                    )
                    if can_adjust:
                        peer_demand_vec = new_demand_vec_peer
                    for serv in peer_res_map.keys():
                        gpu_share = peer_res_map[serv]["gpu"] / len(gpus_realloc)
                        peer_demand_vec_share = [
                            res * gpu_share for res in peer_demand_vec
                        ]
                        peer_res_map[serv] = peer_demand_vec_share

                    for i in range(0, len(rsc_type)):
                        execution_plan[rsc_type[i]][j].extend(
                            [node] * share[i] for node, share in peer_res_map.items()
                        )
            return self._tune(
                job,
                job_infos,
                node_infos,
                demand_vec,
                execution_plan,
                job_gpu_deficit,
                False,
                False,
                True,
                _call_allocate,
                free_rsc,
                fair,
            )

        else:
            raise Exception("Cannot adjust job")

    def allocate_synergy_random(
        self,
        job_infos,
        node_infos,
        free_rsc,
        execution_plan,
        num_gpus,
        job,
        fair=True,
        demand_vec=None,
    ):
        job_demand_vector = job_infos[job].job_demand_vector
        job_demand_vector_gpu_norm = self.gpu_normalized_vector(job_demand_vector)
        rsc_addition, free_rsc_tmp = self._top_synergy_gpus(
            job_demand_vector,
            num_gpus,
            copy.deepcopy(free_rsc),
            job_demand_vector_gpu_norm,
        )
        if rsc_addition is None:
            return False, None, free_rsc
        print("rsc_addition", rsc_addition)
        free_rsc = copy.deepcopy(free_rsc_tmp)
        for i in range(0, len(rsc_type)):
            for node_idx, num in rsc_addition.items():
                execution_plan[rsc_type[i]][job].extend(
                    [node_idx] * int(num) * int(job_demand_vector_gpu_norm[i])
                )
        return True, execution_plan, free_rsc

    def gpu_normalized_vector(self, vector):
        return [item / vector[0] for item in vector]

    def _top_synergy_gpus(
        self, job_demand_vector, num_gpus, free_rsc, job_demand_vector_gpu_norm
    ):
        rsc_addition = {}

        pq = [(num, node_idx) for node_idx, num in free_rsc["nvidia.com/gpu"].items()]
        heapq.heapify(pq)
        is_alloc = False
        while (
            sum(rsc_addition[node_idx] for node_idx in rsc_addition) < num_gpus
            and len(pq) > 0
        ):
            _, gpu_node = heapq.heappop(pq)
            if gpu_node not in rsc_addition:
                rsc_addition[gpu_node] = 0
            if self._fits_in_server(
                gpu_node, num_gpus, job_demand_vector, copy.deepcopy(free_rsc)
            ):
                is_alloc = True
                rsc_addition[gpu_node] += num_gpus
                for i in range(0, len(rsc_type)):
                    free_rsc[rsc_type[i]][gpu_node] -= job_demand_vector[i]
        if not is_alloc:
            pq = [
                (-num, node_idx) for node_idx, num in free_rsc["nvidia.com/gpu"].items()
            ]
            heapq.heapify(pq)
            if self._fits_in_multi_server(
                copy.deepcopy(pq), num_gpus, job_demand_vector, copy.deepcopy(free_rsc)
            ):
                is_alloc = True
                need_alloc_gpu = num_gpus
                while (
                    sum(rsc_addition[node_idx] for node_idx in rsc_addition) < num_gpus
                    and len(pq) > 0
                ):
                    _, node_idx = heapq.heappop(pq)
                    if node_idx not in rsc_addition:
                        rsc_addition[node_idx] = 0
                    free_vector = [free_rsc[rsc][node_idx] for rsc in rsc_type]
                    alloc_gpu = min(free_vector[0], need_alloc_gpu)
                    rsc_addition[node_idx] += int(alloc_gpu)
                    need_alloc_gpu -= alloc_gpu
                    for j in range(0, int(alloc_gpu)):
                        for i in range(0, len(rsc_type)):
                            free_rsc[rsc_type[i]][
                                node_idx
                            ] -= job_demand_vector_gpu_norm[i]

        if sum(rsc_addition[node_idx] for node_idx in rsc_addition) < num_gpus:
            return None, None

        return rsc_addition, free_rsc

    def _fits_in_multi_server(self, pq, num_gpus, job_demand_vector, free_rsc):
        rsc_addition = {}
        while (
            sum(rsc_addition[node_idx] for node_idx in rsc_addition) < num_gpus
            and len(pq) > 0
        ):
            _, node_idx = heapq.heappop(pq)
            if node_idx not in rsc_addition:
                rsc_addition[node_idx] = 0
            free_vector = [free_rsc[rsc][node_idx] for rsc in rsc_type]
            rsc_addition[node_idx] += free_vector[0]
        if sum(rsc_addition[node_idx] for node_idx in rsc_addition) >= num_gpus:
            return True
        return False

    def _fits_in_server(self, node_idx, num_gpus, job_demand_vector, free_rsc):
        free_vector = [free_rsc[rsc][node_idx] for rsc in rsc_type]
        for idx, free_res in enumerate(free_vector):
            required_res = job_demand_vector[idx]
            if free_res < required_res:
                return False
        return True

    def _make_fair_share(self, job_infos, job, demand_vec, node_infos):
        must_switch = False
        new_demand_vec = copy.deepcopy(demand_vec)
        if self.per_server_size_fair == None:
            per_server_size_fair = [
                node_infos[0].resources[rsc] / node_infos[0].resources["nvidia.com/gpu"]
                for rsc in rsc_type
            ]
            self.per_server_size_fair = per_server_size_fair
        for demand, fair in zip(demand_vec, self.per_server_size_fair):
            if demand > fair * job_infos[job].job_gpu_demand:
                must_switch = True
        if must_switch:
            new_demand_vec[1] = (
                self.per_server_size_fair[1] * job_infos[job].job_gpu_demand
            )
            new_demand_vec[2] = (
                self.per_server_size_fair[2] * job_infos[job].job_gpu_demand
            )
            new_demand_vec[3] = (
                self.per_server_size_fair[3] * job_infos[job].job_gpu_demand
            )
            # reset speedup to default
            job.synergy_speedup = 1
            return True, new_demand_vec
        return False, None

    def _get_underutilized_servers(self, job_gpu_deficit, free_rsc):
        nodes_map = {}
        if job_gpu_deficit > 1:
            for node_idx in free_rsc["nvidia.com/gpu"]:
                if free_rsc["nvidia.com/gpu"][node_idx] >= job_gpu_deficit:
                    return {node_idx: job_gpu_deficit}
        pq = [(num, node_idx) for node_idx, num in free_rsc["nvidia.com/gpu"].items()]
        heapq.heapify(pq)
        while job_gpu_deficit > 0 and len(pq) > 0:
            gpus, serv = heapq.heappop(pq)
            if gpus >= job_gpu_deficit:
                nodes_map[serv] = job_gpu_deficit
                return nodes_map
            else:
                nodes_map[serv] = gpus
                job_gpu_deficit -= gpus
        return nodes_map

    def _reallocate_peer(self, demand_vec, avail_vec, node_idx, execution_plan):
        spare_res_need = [max(0, x1 - x2) for (x1, x2) in zip(demand_vec, avail_vec)]
        if all(v == 0 for v in spare_res_need):
            return []

        jobs_to_realloc = []
        job_list = [
            j for j, node in execution_plan["nvidia.com/gpu"].item() if node_idx in node
        ]
        job_list.sort(
            key=lambda x: (
                len(set(execution_plan["nvidia.com/gpu"][x])),
                -sum(len(execution_plan["cpu"][x]))
                / sum(len(execution_plan["nvidia.com/gpu"][x])),
                -sum(len(execution_plan["memory"][x]))
                / sum(len(execution_plan["nvidia.com/gpu"][x])),
            )
        )

        for job in job_list:
            job_gpus_this_server = execution_plan["nvidia.com/gpu"][job].count(node_idx)
            job_fair = [x * job_gpus_this_server for x in self.per_server_size_fair]
            job_alloc_vector = {}
            for rsc in rsc_type:
                job_alloc_vector[rsc] = len(execution_plan[rsc][job])
            job_gpu_share = job_gpus_this_server / job_alloc_vector["nvidia.com/gpu"]
            job_alloc_share = [
                job_alloc_vector[rsc] * job_gpu_share for rsc in job_alloc_vector
            ]
            job_excess_vec = [
                max(0, x1 - x2) for (x1, x2) in zip(job_alloc_share, job_fair)
            ]
            diff = [max(0, x2 - x1) for (x1, x2) in zip(job_excess_vec, spare_res_need)]
            if all(v == 0 for v in diff):
                jobs_to_realloc.append(job)
                return jobs_to_realloc
            elif all(v > 0 for v in diff):
                continue
            else:
                jobs_to_realloc.append(job)
                spare_res_need = diff
        return jobs_to_realloc
