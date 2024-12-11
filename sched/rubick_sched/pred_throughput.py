import autograd
import numpy as np
import collections
import scipy.optimize
import scipy.stats

# Fittable Parameters
PerfParams = collections.namedtuple("PerfParams", [
    "k_bct_fwd",
    "k_c",
    "k_os",
    "k_os_offload",
    "b_overhead",
    "k_off",
    "k_swap",
])

StrategyParams = collections.namedtuple("StrategyParams", [
    "alpha_f",
    "alpha_b",
    "alpha_c",
    "is_mp",
    "is_pp",
    "zero_type",
])


class ModelInfo(object):
    def __init__(self, sequence, hidden_size):
        self.seq = sequence
        self.hs = hidden_size


class ThroughputFunction(object):
    def __init__(self, perf_params, env_params):
        self._perf_params = PerfParams(*perf_params)
        self._env_params = env_params

    def throughput(self, ps, forward_time, num_gpu, local_bsz, num_threads, bandwidth, seq_hid, st_params):
        backward_time = _predict_backward_time(self._perf_params, forward_time)
        network_volume, extra_volume = _predict_network_volume(
            self._perf_params, ps, st_params.is_mp, st_params.is_pp, local_bsz, num_gpu, seq_hid)
        offload_time = _predict_offload_time(ps/num_gpu, st_params.zero_type)
        overlap_time_offload = np.exp(_predict_overlap_sync_time(self._env_params, self._perf_params,
                                                                 backward_time, bandwidth, st_params.alpha_c*network_volume, extra_volume, st_params.zero_type*offload_time, max(st_params.is_mp, st_params.is_pp)))

        optimize_time = _predict_optimize_time(
            self._perf_params, num_gpu, ps, num_threads, st_params.zero_type, max(st_params.is_mp, st_params.is_pp))
        overlap_time_swap = np.exp(_predict_log_overlap_time_swap(
            self._perf_params, optimize_time, st_params.zero_type*offload_time))
        iteration_time = st_params.alpha_f*forward_time + \
            (st_params.alpha_b-1)*backward_time+overlap_time_offload + \
            overlap_time_swap+self._perf_params.b_overhead
        return local_bsz / (iteration_time/1000)


def _predict_optimize_time(params, num_gpus, ps, num_threads, zero_type, not_dp):
    params = PerfParams(*params)
    # Forward/backward passes should scale linearly with the batch size.
    optimize_time = np.select([not_dp == 1, zero_type == 0, zero_type == 1],
                              [params.k_os*ps/num_gpus, params.k_os*ps, params.k_os_offload*ps/(num_gpus*num_threads)], params.k_os*ps/num_gpus)
    return optimize_time


def _predict_backward_time(params, forward_time):
    params = PerfParams(*params)
    # Forward/backward passes should scale linearly with the batch size.
    return params.k_bct_fwd*forward_time


def _predict_overlap_sync_time(env_params, params, backward_time, bandwidth, network_volume, extra_network, offload_time, not_dp):
    params = PerfParams(*params)
    conds = [not_dp == 0]
    con_bw = [bandwidth == 0]
    bandwidth = np.select(con_bw, [30], 1.4)
    sync_time = np.select(conds, [np.exp(_predict_log_overlap_time_backward(
        params, network_volume/bandwidth, backward_time))], network_volume/bandwidth+backward_time+extra_network/30)
    conds = [offload_time == 0]
    overlap_sync_time = np.select(conds, [np.log(
        sync_time)], _predict_log_overlap_time_offload(params, sync_time, offload_time))
    return overlap_sync_time


def _predict_log_overlap_time_backward(params, comm_time, backward_time):
    gamma = params.k_c
    return np.log(comm_time ** gamma + backward_time ** gamma) / gamma


def _predict_log_overlap_time_offload(params, sync_time, offload_time):
    gamma = params.k_off
    return np.log(sync_time ** gamma + offload_time ** gamma) / gamma


def _predict_log_overlap_time_swap(params, optimize_time, offload_time):
    gamma = params.k_swap
    return np.log(optimize_time ** gamma + offload_time ** gamma) / gamma


def _predict_network_volume(params, ps, is_mp, is_pp, bs, num_gpus, seq_hid=1):
    params = PerfParams(*params)
    conds = [is_pp == 1, is_mp == 0]
    network_volume = np.select(conds, [2*8*seq_hid/1024/1024, 4*2*ps*(
        num_gpus-1)/num_gpus], 4*8*bs*seq_hid*(num_gpus-1)/num_gpus/1024/1024)
    conds = [is_pp == 1]
    extra_network = np.select(conds, [2*8*(num_gpus-1)*seq_hid/1024/1024], 0)
    return network_volume, extra_network


def _predict_offload_time(optimize_ps, zero_type):
    pcie_bandwidth = 1  # 5GB/s=5MB/ms
    conds = [zero_type == 1]
    return np.select(conds, [zero_type*4*optimize_ps/pcie_bandwidth], 0)


def _predict_iteration_time(str_params, forward_time, pred_backward, pred_log_overlap_sync, pred_log_overlap_swap, params):
    params = PerfParams(*params)
    return str_params[0]*forward_time+(np.array(str_params[1])-np.array([1]*len(str_params[1])))*pred_backward+np.exp(pred_log_overlap_sync)+np.exp(pred_log_overlap_swap)+np.array([params.b_overhead]*len(str_params[1]))


def fit_perf_params(offload_params, env_params, num_nodes, num_gpus, ps, batch_size,
                    forward_time, overlap_sync,
                    optimize_time, iteration_time,
                    seq_hid, num_threads,
                    zero_type, is_mp, is_pp, str_params):

    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    num_nodes = np.array(num_nodes)
    num_gpus = np.array(num_gpus)

    forward_time = np.array(forward_time)
    overlap_sync = np.array(overlap_sync)
    optimize_time = np.array(optimize_time)
    iteration_time = np.array(iteration_time)

    bs = np.array(batch_size)
    seq_hid = np.array(seq_hid)
    num_threads = np.array(num_threads)

    str_params = np.transpose(str_params).tolist()
    # Set initial params to reasonable values.
    params = [1.50, 0.1, 0.1, 1, 8, 1, 1]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    lower = [1, 1]+[1e-8] * 3+[1, 1]
    upper = [10]+[np.inf] * 6
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True)
    args = (offload_params, env_params, num_nodes, num_gpus, ps, batch_size,
            forward_time, overlap_sync,
            optimize_time, iteration_time,
            seq_hid, num_threads,
            zero_type, is_mp, is_pp, str_params)
    grad_fn = autograd.grad(_obj_fn)
    result = scipy.optimize.minimize(_obj_fn, params, args=args,
                                     jac=grad_fn, bounds=bounds)
    params = result.x
    np = orig_np  # Restore original numpy.
    return PerfParams(*params)


def _rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())


def _obj_fn(params, offload_params, env_params, num_nodes, num_gpus, ps, batch_size,
            forward_time, overlap_sync,
            optimize_time, iteration_time,
            seq_hid, num_threads,
            zero_type, is_mp, is_pp, str_params):

    params = PerfParams(*params)

    # Backward time
    pred_backward = _predict_backward_time(params, forward_time)

    # All-reduce extra communication time
    pred_network, extra_network = _predict_network_volume(
        params, ps, is_mp, is_pp, batch_size, num_gpus, seq_hid)
    pred_offload_time = _predict_offload_time(ps/num_gpus, zero_type)
    pred_log_overlap_sync = _predict_overlap_sync_time(
        env_params, params, pred_backward, 0, pred_network, extra_network, pred_offload_time, np.maximum(is_mp, is_pp))

    # Optimizer time
    pred_optimize_time = _predict_optimize_time(
        params, num_gpus, ps, num_threads, zero_type, np.maximum(is_mp, is_pp))
    pred_log_overlap_swap = _predict_log_overlap_time_swap(
        params, pred_optimize_time, pred_offload_time)
    pred_iteration_time = _predict_iteration_time(
        str_params, forward_time, pred_backward, pred_log_overlap_sync, pred_log_overlap_swap, params)

    # RMSLError of optim step time predictions.
    backward_all = pred_backward * \
        (str_params[0]-np.array([1]*len(str_params[0]))) + \
        np.exp(pred_log_overlap_sync)
    err2 = _rmse(np.log(backward_all), np.log(overlap_sync))

    err3 = _rmse((pred_log_overlap_swap), np.log(optimize_time))

    err4 = _rmse(np.log(pred_iteration_time), np.log(iteration_time))

    return err2+err3+err4
