import collections
from model import Model
import copy
import math
rsc_type = ["gpu", "cpu", "bandwidth"]
empty_rsc = {"gpu": 0, "cpu": 0, "bandwidth": 0}


class RubickPolicy(object):
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        self._status = {}
        self._queue = []

    def optimize(self, job_infos, node_infos, prev_execution_plan, node_template):
        self._status = {key: val for key,
                        val in self._status.items() if key in job_infos}
        self._queue = [key for key in self._queue if key in job_infos]
        execution_plan = {}
        job_alloc = {}
        total_rsc = {}
        free_rsc = {}
        node_total_rsc = {}

        # Add new jobs to pending.
        for key, job in job_infos.items():
            if key not in self._status:
                self._status[key] = 'PENDING'
                self._queue.append(key)
            if self._status[key] == 'RUNNING' and key in self._queue:
                self._queue.pop(self._queue.index(key))
        for rsc in rsc_type:
            execution_plan[rsc] = {
                k: v for k, v in prev_execution_plan[rsc].items() if k in job_infos}
            total_rsc[rsc] = {idx: int(node.resources[rsc])
                              for idx, node in node_infos.items()}
            free_rsc[rsc] = collections.Counter(
                total_rsc[rsc]) - collections.Counter(sum(execution_plan[rsc].values(), []))
        job_alloc = self.get_job_alloc(execution_plan)

        execution_plan["strategy"] = {
            k: v for k, v in prev_execution_plan["strategy"].items() if k in job_infos}

        print("-----------start optimize--------")
        for job in self._queue:
            if job not in job_alloc:
                job_alloc[job] = {}
            for rsc in rsc_type:
                execution_plan[rsc][job] = []
                job_alloc[job][rsc] = 0
                free_rsc[rsc] = collections.Counter(
                    total_rsc[rsc]) - collections.Counter(sum(execution_plan[rsc].values(), []))
            is_allocated = False

            max_throughput_config = job_infos[job].max_throughput_config
            max_throughput_norm_demand_vector = self.gpu_normalized_vector(
                max_throughput_config[:2])
            max_throughput_norm_demand_vector.append(max_throughput_config[2])
            single_node = False
            # Allocate on a single node
            if max_throughput_norm_demand_vector[-1] == 0:
                single_node = True
            max_throughput_rsc = self._max_throughput_gpus(
                max_throughput_norm_demand_vector, max_throughput_config[0], copy.deepcopy(free_rsc), single_node)
            if max_throughput_rsc:
                is_allocated = True
                for i in range(0, len(rsc_type)):
                    for node_idx, num in max_throughput_rsc.items():
                        if i == 2 and not single_node:
                            num = 1
                            job_alloc[job][rsc_type[i]] = num * \
                                int(max_throughput_norm_demand_vector[i])
                        else:
                            job_alloc[job][rsc_type[i]] += num * \
                                int(max_throughput_norm_demand_vector[i])
                        execution_plan[rsc_type[i]][job].extend(
                            [node_idx]*num*int(max_throughput_norm_demand_vector[i]))
                execution_plan["strategy"][job] = max_throughput_config[-1]
                if len([v for k, v in max_throughput_rsc.items() if v > 0]) > 1:
                    job_infos[job].is_distributed = True
            # Step2: shrink
            if not is_allocated:
                print("----------start shrink----------")
                max_throughput = 0
                max_execution_plan = copy.deepcopy(execution_plan)
                for node_idx in node_infos:
                    targeted_nodes, node_total_rsc = self.get_targeted_nodes(
                        execution_plan, node_idx, node_infos)
                    is_allocated, tmp_execution_plan, shrink_throughput = self.shrink(
                        job_infos, job, targeted_nodes, node_idx, node_total_rsc, copy.deepcopy(execution_plan), copy.deepcopy(job_alloc))
                    if is_allocated and shrink_throughput > max_throughput:
                        max_throughput = shrink_throughput
                        max_execution_plan = tmp_execution_plan
                execution_plan = max_execution_plan
                job_alloc = self.get_job_alloc(execution_plan)
            if is_allocated:
                self._status[job] = "RUNNING"

        print("----------start expand----------")
        for node_idx in node_infos:
            targeted_nodes, node_total_rsc = self.get_targeted_nodes(
                execution_plan, node_idx, node_infos)
            execution_plan = self.expand(job_infos, copy.deepcopy(
                targeted_nodes), node_idx, node_total_rsc, execution_plan)

        return execution_plan, 0

    def shrink(self, job_infos, targeted_job, targeted_nodes, node_idx, total_rsc, execution_plan, job_alloc):

        throughput_node = 0
        free_rsc = {}
        for rsc in rsc_type:
            free_rsc[rsc] = total_rsc[rsc] - \
                (sum(r[rsc] for job, r in targeted_nodes.items()))
        if job_infos[targeted_job].job_gpu_demand > 8:
            bw = free_rsc["bandwidth"]
        else:
            bw = 0
        if free_rsc["gpu"] < 0:
            job_throughput_base, job_strategy, job_rsc_alloc = -1, None, None
        else:
            job_throughput_base, job_strategy, job_rsc_alloc = job_infos[targeted_job].rsc_matrix[
                free_rsc["gpu"]][free_rsc["cpu"]//(1e5 if free_rsc["gpu"] <= 0 else free_rsc["gpu"])][bw]

        throughput_slope_down = 0
        shrink_job = None
        # Find the maximun delta throughput
        max_delta_throughput = job_throughput_base / \
            job_infos[targeted_job].origin_tpt-throughput_slope_down
        max_rsc_shrink = None
        max_targeted_job_rsc = copy.deepcopy(job_rsc_alloc)
        max_job_strategy = copy.deepcopy(job_strategy)

        can_allocated = False
        delta_addition = True
        should_exit = False
        while not can_allocated and not should_exit:
            delta_addition = True
            rsc_delta = [copy.deepcopy(free_rsc), copy.deepcopy(free_rsc)]

            if job_infos[targeted_job].job_gpu_demand > 8:
                rsc_delta.append(copy.deepcopy(free_rsc))
            while delta_addition:
                rsc_delta[0]["gpu"] += 1
                rsc_delta[1]["cpu"] = rsc_delta[1]["cpu"] if rsc_delta[1]["gpu"] == 0 else min(
                    rsc_delta[1]["cpu"]+rsc_delta[1]["gpu"], total_rsc["cpu"])
                if job_infos[targeted_job].job_gpu_demand > 8:
                    rsc_delta[2]["bandwidth"] = max(
                        rsc_delta[2]["bandwidth"], rsc_delta[2]["bandwidth"]+1)
                else:
                    rsc_delta[0]["bandwidth"] = 0
                    rsc_delta[1]["bandwidth"] = 0
                if rsc_delta[0]["gpu"] >= total_rsc["gpu"] and (rsc_delta[1]["cpu"] >= total_rsc["cpu"] or rsc_delta[1]["gpu"] == 0):
                    delta_addition = False
                    should_exit = True
                    break
                rsc_delta.sort(key=lambda r: job_infos[targeted_job].rsc_matrix[r["gpu"]][r["cpu"]//(
                    1e5 if r["gpu"] == 0 else r["gpu"])][r["bandwidth"]][0], reverse=True)
                # Targeted job and resource
                for rsc_delta_target in rsc_delta:
                    job_throughput, job_strategy, job_rsc_alloc = job_infos[targeted_job].rsc_matrix[rsc_delta_target["gpu"]][rsc_delta_target["cpu"]//(
                        1e5 if rsc_delta_target["gpu"] == 0 else rsc_delta_target["gpu"])][rsc_delta_target["bandwidth"]]
                    if job_throughput == -1:
                        if rsc_delta_target == rsc_delta[0]:
                            delta_addition = True
                        break
                    else:
                        rsc_delta_matrix = {
                            rsc: max(rsc_delta_target[rsc]-free_rsc[rsc], 0) for rsc in rsc_type}
                        # Shrink resources of other jobs on the machine
                        throughput_slope_down, rsc_shrink, job_rsc_alloc, job_strategy = self.shrink_delta_rsc(
                            job_infos, rsc_delta_matrix, copy.deepcopy(targeted_nodes), copy.deepcopy(job_alloc), targeted_job, free_rsc)  # use cpu
                        if throughput_slope_down+1e-2 >= 1e6:
                            continue
                        # Satisfy shrink principle
                        if throughput_slope_down > max_delta_throughput:
                            max_delta_throughput = throughput_slope_down
                            delta_addition = False
                            can_allocated = True
                            max_rsc_shrink = copy.deepcopy(rsc_shrink)
                            max_targeted_job_rsc = copy.deepcopy(job_rsc_alloc)
                            max_job_strategy = copy.deepcopy(job_strategy)
                            break
        if max_delta_throughput >= 0 and max_rsc_shrink != {}:
            execution_plan = self.execution_plan_adapt(
                max_rsc_shrink, execution_plan, node_idx, targeted_job, max_targeted_job_rsc, max_job_strategy, targeted_nodes)
            return True, execution_plan, max_delta_throughput
        else:
            return False, None, None

    def execution_plan_adapt(self, rsc_shrink, execution_plan, node_idx, targeted_job, targeted_job_rsc, targeted_job_strategy, targeted_nodes):
        if rsc_shrink != None:
            execution_plan = self.remove_from_execution_plan(
                execution_plan, rsc_shrink, node_idx, targeted_nodes)
        for rsc in rsc_type:
            execution_plan[rsc][targeted_job] = []
            if rsc == "cpu":
                execution_plan[rsc][targeted_job].extend(
                    [node_idx]*targeted_job_rsc[rsc]*targeted_job_rsc["gpu"])
            else:
                execution_plan[rsc][targeted_job].extend(
                    [node_idx]*targeted_job_rsc[rsc])
        execution_plan["strategy"][targeted_job] = targeted_job_strategy
        return execution_plan

    def remove_from_execution_plan(self, execution_plan, rsc_shrink, node_idx, targeted_nodes):

        for rsc in rsc_type:
            for s_j in rsc_shrink:
                shrink_num = rsc_shrink[s_j][rsc]
                i = 0
                while i < min(shrink_num, targeted_nodes[s_j][rsc]):
                    i += 1
                    if rsc_type == "bandwidth":
                        for i in list(collections.Counter(execution_plan[rsc][s_j]).keys()):
                            execution_plan[rsc][s_j].remove(i)
                    else:
                        execution_plan[rsc][s_j].remove(node_idx)
                while i < shrink_num:
                    i += 1
                    execution_plan[rsc][s_j].remove(
                        list(collections.Counter(execution_plan[rsc][s_j]).keys())[-1])
                execution_plan["strategy"][s_j] = rsc_shrink[s_j]["strategy"]

        execution_plan = self.clear_bw(execution_plan)
        return execution_plan

    def clear_bw(self, execution_plan):
        node_idx = set()
        for j in execution_plan["gpu"]:
            node_idx = set(
                execution_plan["gpu"][j]) | set(execution_plan["cpu"][j])
            rm_node_idx = set(execution_plan["bandwidth"][j]) - node_idx
            bw_cnt = collections.Counter(execution_plan["bandwidth"][j])
            for node in rm_node_idx:
                for i in range(0, bw_cnt[node]):
                    execution_plan["bandwidth"][j].remove(node)
        return execution_plan

    def sort_slope_v(self, slope_v, shrink_rsc, job_infos, targeted_job, free_rsc, targeted_nodes):
        targeted_job_rsc_alloc = {}
        targeted_job_strategy = {}
        for k, v in slope_v.items():
            updated_rsc_matrix = {
                rsc: free_rsc[rsc]+min(shrink_rsc[k][rsc], targeted_nodes[k][rsc]) for rsc in rsc_type}
            updated_rsc_matrix["bandwidth"] = 0
            job_throughput, targeted_job_strategy[k], targeted_job_rsc_alloc[k] = job_infos[targeted_job].rsc_matrix[updated_rsc_matrix["gpu"]][updated_rsc_matrix["cpu"]//
                (1e5 if updated_rsc_matrix["gpu"] == 0 else updated_rsc_matrix["gpu"])][updated_rsc_matrix["bandwidth"]]
            slope_v[k] = job_throughput / \
                job_infos[targeted_job].origin_tpt-v/job_infos[k].origin_tpt
        return sorted(slope_v.items(), key=lambda kv: (kv[1], kv[0]), reverse=True), targeted_job_rsc_alloc, targeted_job_strategy

    def shrink_delta_rsc(self, job_infos, rsc_delta_target, targeted_nodes, job_alloc, targeted_job, free_rsc):
        # Format of slope_matrix: {job_name,througput_slope_down}
        slope_matrix = {}
        slope_down = 0
        rsc_shrink = {}
        shrink_rsc = {}
        if rsc_delta_target["gpu"] > 0:
            for job in targeted_nodes:
                rsc = [job_alloc[job]["gpu"], job_alloc[job]["cpu"] //
                       job_alloc[job]["gpu"], job_alloc[job]["bandwidth"]]
                num_restarts = self.judge_restart(job_infos[job])
                slope_matrix[job], shrink_rsc[job] = self.cal_slope(
                    "gpu", rsc[0], rsc[1], rsc[2], job_infos[job].rsc_matrix, rsc_delta_target["gpu"], job_infos[job].local_bsz, num_restarts)
            slope_v = {k: v for k, v in slope_matrix.items() if v}
            if len(slope_v) == 0:
                return slope_down, rsc_shrink, None, None
            slope_v, targeted_job_rsc_alloc, targeted_job_strategy = self.sort_slope_v(
                slope_v, shrink_rsc, job_infos, targeted_job, free_rsc, targeted_nodes)
            slope_down += slope_v[0][1]
            shrink_job_name = slope_v[0][0]
            if shrink_rsc[shrink_job_name]["gpu"] <= 0:
                return slope_down, rsc_shrink, None, None
            rsc_shrink[shrink_job_name] = copy.deepcopy(
                shrink_rsc[shrink_job_name])
            job_alloc[shrink_job_name]["cpu"] -= shrink_rsc[shrink_job_name]["cpu"]
            job_alloc[shrink_job_name]["gpu"] -= shrink_rsc[shrink_job_name]["gpu"]
            job_alloc[shrink_job_name]["bandwidth"] -= shrink_rsc[shrink_job_name]["bandwidth"]
        elif rsc_delta_target["cpu"] > 0:
            for job in targeted_nodes:
                if job_infos[job].is_distributed:
                    continue
                num_restarts = self.judge_restart(job_infos[job])
                rsc = [targeted_nodes[job]["gpu"], targeted_nodes[job]["cpu"] //
                       targeted_nodes[job]["gpu"], targeted_nodes[job]["bandwidth"]]
                slope_matrix[job], shrink_rsc[job] = self.cal_slope(
                    "cpu", rsc[0], rsc[1], rsc[2], job_infos[job].rsc_matrix, rsc_delta_target["cpu"], job_infos[job].local_bsz, num_restarts)
            slope_v = {k: v for k, v in slope_matrix.items() if v}
            if len(slope_v) == 0:
                return slope_down, rsc_shrink, None, None
            slope_v, targeted_job_rsc_alloc, targeted_job_strategy = self.sort_slope_v(
                slope_v, shrink_rsc, job_infos, targeted_job, free_rsc, targeted_nodes)
            slope_down += slope_v[0][1]
            shrink_job_name = slope_v[0][0]
            if shrink_rsc[shrink_job_name]["cpu"] <= 0:
                return slope_down, rsc_shrink, None, None
            rsc_shrink[shrink_job_name] = copy.deepcopy(
                shrink_rsc[shrink_job_name])
            targeted_nodes[shrink_job_name]["cpu"] -= shrink_rsc[shrink_job_name]["cpu"]
            targeted_nodes[shrink_job_name]["gpu"] -= shrink_rsc[shrink_job_name]["gpu"]
        elif rsc_delta_target["bandwidth"] > 0:
            for job in targeted_nodes:
                if job_infos[job].is_distributed:
                    continue
                num_restarts = self.judge_restart(job_infos[job])
                rsc = [targeted_nodes[job]["gpu"], targeted_nodes[job]["cpu"] //
                       targeted_nodes[job]["gpu"], targeted_nodes[job]["bandwidth"]]
                slope_matrix[job], shrink_rsc[job] = self.cal_slope(
                    "gpu", rsc[0], rsc[1], rsc[2], job_infos[job].rsc_matrix, rsc_delta_target["bandwidth"], job_infos[job].local_bsz, num_restarts)
            slope_v, targeted_job_rsc_alloc, targeted_job_strategy = {
                k: v for k, v in slope_matrix.items() if v}
            if len(slope_v) == 0:
                return slope_down, rsc_shrink, None, None
            slope_v = self.sort_slope_v(
                slope_v, shrink_rsc, job_infos, targeted_job, free_rsc, targeted_nodes)
            slope_down += slope_v[0][1]
            shrink_job_name = slope_v[0][0]
            if shrink_rsc[shrink_job_name]["bandwidth"] <= 0:
                return slope_down, rsc_shrink, None, None
            rsc_shrink[shrink_job_name] = copy.deepcopy(
                shrink_rsc[shrink_job_name])
            targeted_nodes[shrink_job_name]["cpu"] -= shrink_rsc[shrink_job_name]["cpu"]
            targeted_nodes[shrink_job_name]["gpu"] -= shrink_rsc[shrink_job_name]["gpu"]
            rsc_shrink["bandwidth"] += 1
        return slope_down, rsc_shrink, targeted_job_rsc_alloc[shrink_job_name], targeted_job_strategy[shrink_job_name]

    def expand(self, job_infos, targeted_nodes, node_idx, total_rsc, execution_plan):
        if targeted_nodes == {}:
            return execution_plan
        # Format of free_rsc: {gpu:xx,cpu:xx,bw:xx}
        free_rsc = {}
        for rsc in rsc_type:
            free_rsc[rsc] = total_rsc[rsc] - \
                (sum(r[rsc] for job, r in targeted_nodes.items()))
        for rsc in rsc_type:
            if free_rsc[rsc]/total_rsc[rsc] > 0.5:
                rsc_delta_matrix = copy.deepcopy(empty_rsc)
                rsc_delta_matrix[rsc] = max(free_rsc[rsc], 0)
                expand_job, rsc_addition = self.expand_delta_rsc(
                    job_infos, rsc_delta_matrix, targeted_nodes)
                if len(expand_job) == 0:
                    break
                for job in expand_job:
                    free_rsc = {
                        rsc: free_rsc[rsc]-rsc_addition[job][rsc] for rsc in rsc_type}
                    targeted_nodes[job] = {
                        rsc: targeted_nodes[job][rsc]+rsc_addition[job][rsc] for rsc in rsc_type}
                    for add_rsc in rsc_addition[job]:
                        if add_rsc == "strategy":
                            execution_plan[add_rsc][job] = rsc_addition[job][add_rsc]
                        else:
                            execution_plan[add_rsc][job].extend(
                                [node_idx]*rsc_addition[job][add_rsc])
        return execution_plan

    def expand_delta_rsc(self, job_infos, rsc_delta_target, targeted_nodes):
        # Format of slope_matrix: {job_name,througput_slope_down}
        slope_matrix = {}
        expand_job = []
        expand_rsc = {}
        rsc_expand = {}
        if rsc_delta_target["gpu"] > 0:
            delta_gpu = rsc_delta_target["gpu"]
            while sum(rsc_expand[j]["gpu"] for j in rsc_expand) < delta_gpu:
                for job in targeted_nodes:
                    if job_infos[job].steps == 0 or job_infos[job].is_distributed:
                        continue
                    num_restarts = self.judge_restart(job_infos[job])
                    # Original resource allocaiton
                    rsc = [targeted_nodes[job]["gpu"], targeted_nodes[job]["cpu"] //
                           (targeted_nodes[job]["gpu"]), targeted_nodes[job]["bandwidth"]]
                    slope_matrix[job], expand_rsc[job] = self.cal_slope(
                        "gpu", rsc[0], rsc[0], rsc[2], job_infos[job].rsc_matrix, -rsc_delta_target["gpu"], job_infos[job].local_bsz, num_restarts)
                if slope_matrix == {}:
                    return expand_job, rsc_expand
                slope_v = {k: v for k, v in slope_matrix.items() if v}
                slope_v = sorted(
                    slope_v.items(), key=lambda kv: (kv[1], kv[0]))
                if len(slope_v) == 0:
                    return expand_job, rsc_expand
                for item in slope_v:
                    expand_job_name = item[0]
                    if expand_rsc[expand_job_name]["gpu"] <= 0:
                        return expand_job, rsc_expand
                    if -item[1] > 1e-2:
                        expand_job.append(expand_job_name)
                        rsc_expand[expand_job_name] = copy.deepcopy(
                            expand_rsc[expand_job_name])
                        targeted_nodes[expand_job_name]["cpu"] += expand_rsc[expand_job_name]["cpu"]
                        targeted_nodes[expand_job_name]["gpu"] += expand_rsc[expand_job_name]["gpu"]
                        rsc_delta_target["gpu"] -= expand_rsc[expand_job_name]["gpu"]
                        break
                    if item == slope_v[len(slope_v)-1]:
                        return expand_job, rsc_expand
        elif rsc_delta_target["cpu"] > 0:
            delta_cpu = rsc_delta_target["cpu"]
            while sum(rsc_expand[j]["cpu"] for j in rsc_expand) < delta_cpu:
                for job in targeted_nodes:
                    if job_infos[job].steps == 0:
                        continue
                    # Original resource allocaiton
                    rsc = [targeted_nodes[job]["gpu"], targeted_nodes[job]["cpu"] //
                           (targeted_nodes[job]["gpu"]), targeted_nodes[job]["bandwidth"]]
                    num_restarts = self.judge_restart(job_infos[job])
                    slope_matrix[job], expand_rsc[job] = self.cal_slope(
                        "cpu", rsc[0], rsc[1], rsc[2], job_infos[job].rsc_matrix, -rsc_delta_target["cpu"], job_infos[job].local_bsz, num_restarts)
                if slope_matrix == {}:
                    return expand_job, rsc_expand
                slope_v = {k: v for k, v in slope_matrix.items() if v}
                slope_v = sorted(
                    slope_v.items(), key=lambda kv: (kv[1], kv[0]))
                if len(slope_v) == 0:
                    return expand_job, rsc_expand
                for item in slope_v:
                    expand_job_name = item[0]
                    if expand_rsc[expand_job_name]["cpu"] <= 0:
                        return expand_job, rsc_expand
                    if -item[1] > 1e-2:
                        expand_job.append(expand_job_name)
                        rsc_expand[expand_job_name] = copy.deepcopy(
                            expand_rsc[expand_job_name])
                        targeted_nodes[expand_job_name]["cpu"] += expand_rsc[expand_job_name]["cpu"]
                        targeted_nodes[expand_job_name]["gpu"] += expand_rsc[expand_job_name]["gpu"]
                        rsc_delta_target["cpu"] -= expand_rsc[expand_job_name]["cpu"]
                        break
                    if item == slope_v[len(slope_v)-1]:
                        return expand_job, rsc_expand
        elif rsc_delta_target["bandwidth"] > 0:
            for job in targeted_nodes:
                if job_infos[job].steps == 0:
                    continue
                num_restarts = self.judge_restart(job_infos[job])
                rsc = [targeted_nodes[job]["gpu"], targeted_nodes[job]["cpu"] //
                       (targeted_nodes[job]["gpu"]), targeted_nodes[job]["bandwidth"]]
                slope_matrix[job], expand_rsc[job] = self.cal_slope(
                    "bandwidth", rsc[0], rsc[1], rsc[2], job_infos[job].rsc_matrix, -rsc_delta_target["bandwidth"], job_infos[job].local_bsz, num_restarts)
            slope_v = {k: v for k, v in slope_matrix.items() if v}
            slope_v = sorted(slope_v.items(), key=lambda kv: (kv[1], kv[0]))
            if len(slope_v) <= 0 or slope_v[0][1]+1e-2 >= 1e6:
                return [], None
            for item in slope_v:
                expand_job_name = item[0]
                if expand_rsc[expand_job_name]["bandwidth"] <= 0:
                    return expand_job, rsc_expand
                if -item[1] > 1e-2:
                    rsc_expand["bandwidth"] = 1
                    expand_job.append(slope_v[0][0])
                    break
        return expand_job, rsc_expand

    # Slope definition: [throughput(n_rsc)-throughpput(n_rsc-delta)]/delta
    # Thus, slope(min_n_rsc)=0
    # value=0: No influence when scaling down.
    # value>0: The larger the value, the greater the influence of /delta rsc.
    # value=1e6: Tesource failed to (re)-allocation.
    def cal_slope(self, t_rsc_type, gpu_num, cpu_num, bandwidth, rsc_matrix, delta_rsc, local_bsz, restarts):
        penalty_overhead = 0
        rsc_shrink = {}
        if t_rsc_type == "gpu":
            if rsc_matrix[gpu_num-delta_rsc][cpu_num][bandwidth][0] == -1:
                return None, None
            original_rsc_matrix = rsc_matrix[gpu_num][cpu_num][bandwidth]
            changed_rsc_matrix = rsc_matrix[gpu_num -
                                            delta_rsc][cpu_num][bandwidth]
            rsc_shrink["gpu"] = abs(
                original_rsc_matrix[2]["gpu"]-changed_rsc_matrix[2]["gpu"])
            rsc_shrink["cpu"] = abs(original_rsc_matrix[2]["gpu"]*original_rsc_matrix[2]
                                    ["cpu"]-changed_rsc_matrix[2]["gpu"]*changed_rsc_matrix[2]["cpu"])
            rsc_shrink["bandwidth"] = abs(
                original_rsc_matrix[2]["bandwidth"]-changed_rsc_matrix[2]["bandwidth"])
            rsc_shrink["strategy"] = changed_rsc_matrix[1]
            return original_rsc_matrix[0]-local_bsz/(local_bsz/changed_rsc_matrix[0]+restarts*penalty_overhead), rsc_shrink
        elif t_rsc_type == "cpu":
            if rsc_matrix[gpu_num][cpu_num-delta_rsc][bandwidth][0] == -1:
                return None, None
            original_rsc_matrix = rsc_matrix[gpu_num][cpu_num][bandwidth]
            changed_rsc_matrix = rsc_matrix[gpu_num][cpu_num -
                                                     delta_rsc][bandwidth]
            rsc_shrink["gpu"] = abs(
                original_rsc_matrix[2]["gpu"]-changed_rsc_matrix[2]["gpu"])
            rsc_shrink["cpu"] = abs(original_rsc_matrix[2]["gpu"]*original_rsc_matrix[2]
                                    ["cpu"]-changed_rsc_matrix[2]["gpu"]*changed_rsc_matrix[2]["cpu"])
            rsc_shrink["bandwidth"] = abs(
                original_rsc_matrix[2]["bandwidth"]-changed_rsc_matrix[2]["bandwidth"])
            rsc_shrink["strategy"] = changed_rsc_matrix[1]
            return original_rsc_matrix[0]-local_bsz/(local_bsz/changed_rsc_matrix[0]+restarts*penalty_overhead), rsc_shrink
        elif t_rsc_type == "bandwidth":
            if rsc_matrix[gpu_num][cpu_num][bandwidth-delta_rsc][0] == -1:
                return None, None
            original_rsc_matrix = rsc_matrix[gpu_num][cpu_num][bandwidth]
            changed_rsc_matrix = rsc_matrix[gpu_num][cpu_num][bandwidth-delta_rsc]
            rsc_shrink["gpu"] = abs(
                original_rsc_matrix[2]["gpu"]-changed_rsc_matrix[2]["gpu"])
            rsc_shrink["cpu"] = abs(original_rsc_matrix[2]["gpu"]*original_rsc_matrix[2]
                                    ["cpu"]-changed_rsc_matrix[2]["gpu"]*changed_rsc_matrix[2]["cpu"])
            rsc_shrink["bandwidth"] = abs(
                original_rsc_matrix[2]["bandwidth"]-changed_rsc_matrix[2]["bandwidth"])
            rsc_shrink["strategy"] = changed_rsc_matrix[1]
            return original_rsc_matrix[0]-local_bsz/(local_bsz/changed_rsc_matrix[0]+restarts*penalty_overhead), rsc_shrink

    def judge_restart(self, job_info):
        if job_info.restarts == None:
            return 1
        else:
            return job_info.restarts+1

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

    def _max_throughput_gpus(self, norm_demand_vector, num_gpus, free_rsc, single_node):
        rsc_alloc = {}
        free_gpu = sorted(
            free_rsc["gpu"].items(),key=lambda x: x[1], reverse=True)
        for f_g in free_gpu:
            gpu_node = f_g[0]
            if single_node:
                rsc_alloc = {}
            rsc_alloc[gpu_node] = 0
            gpus = free_rsc["gpu"][gpu_node]
            for gpu in range(0, gpus):
                if self._fits_in_nodes(gpu_node, norm_demand_vector, free_rsc):
                    rsc_alloc[gpu_node] += 1
                    free_rsc["gpu"][gpu_node] -= 1
                    free_rsc["cpu"][gpu_node] -= norm_demand_vector[1]
                    if not single_node and rsc_alloc[gpu_node] == 1:
                        free_rsc["bandwidth"][gpu_node] -= norm_demand_vector[2]
                if sum(rsc_alloc[node_idx] for node_idx in rsc_alloc) == num_gpus:
                    return rsc_alloc
        return None

    def _fits_in_nodes(self, node_idx, norm_demand_vector, free_rsc):
        free_vector = [free_rsc[rsc][node_idx] for rsc in rsc_type]
        for idx, free_res in enumerate(norm_demand_vector):
            required_res = norm_demand_vector[idx]
            if free_res < required_res:
                return False
        return True

    def gpu_normalized_vector(self, vector):
        vector[1] *= vector[0]
        return [item/vector[0] for item in vector]

    def get_job_alloc(self, execution_plan):
        job_alloc = {}
        for rsc in rsc_type:
            for k, v in execution_plan[rsc].items():
                if k not in job_alloc:
                    job_alloc[k] = {}
                job_alloc[k][rsc] = len(
                    v)/(max(1, len(set(execution_plan[rsc][k]))) if rsc == "bandwidth" else 1)
        return job_alloc
