import os
import pandas

from scipy.interpolate import interp1d, LinearNDInterpolator


def memoize(f):
    memo = {}

    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]

    return helper


class Application(object):
    def __init__(
        self,
        trace_dir,
        init_batch_size=None,
        max_batch_size=None,
        min_local_bsz=None,
        max_local_bsz=None,
        max_steps=None,
        target_metric=None,
    ):
        self.name = os.path.basename(trace_dir)
        self.placements = pandas.read_csv(os.path.join(trace_dir, "placements.csv"))
        self.placements["num_nodes"] = self.placements.placement.apply(
            lambda p: len(str(p))
        )
        self.placements["num_gpus"] = self.placements.placement.apply(
            lambda p: sum(map(int, str(p)))
        )

        self.max_steps = max_steps
        self.init_batch_size = 64  # init_batch_size or min(self.validation)
        self.max_batch_size = 64  # max_batch_size or max(self.validation)
        self.min_local_bsz = 64  # min_local_bsz or self.placements.local_bsz.min()
        self.max_local_bsz = 64  # max_local_bsz or self.placements.local_bsz.max()

    @memoize
    def get_progress(self, epoch):
        if epoch == 0:
            return 0.0
        return 1

    @memoize
    def get_completion_steps(self, duration, placement, exec_plan):
        exec_plan = [exec_plan]
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i] for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        if placement_id in self.placements.placement.values:
            if exec_plan in [["zero-offload"], ["zero-dp"], ["ga"], ["gc"]]:
                df = self.placements[
                    (self.placements["placement"] == placement_id)
                    & (self.placements["exec_plan"].isin(exec_plan))
                ]
            else:
                for i in self.placements["exec_plan"]:
                    if type(i) == type(1):
                        df = self.placements[
                            (self.placements["placement"] == placement_id)
                            & (self.placements["exec_plan"] == int(exec_plan[0]))
                        ]
                    else:
                        df = self.placements[
                            (self.placements["placement"] == placement_id)
                            & (self.placements["exec_plan"].isin(exec_plan))
                        ]
                    break
            if len(df) == 0:
                df = self.placements[
                    (self.placements["placement"] == placement_id)
                    & (self.placements["exec_plan"].isin(["ga"]))
                ]
            return round(duration / df["iter_time"].values[0])
        if exec_plan not in [["zero-offload"], ["zero-dp"], ["ga"], ["gc"]]:
            for i in self.placements["exec_plan"]:
                if type(i) == type(1):
                    df = self.placements[
                        (self.placements["exec_plan"] == int(exec_plan[0]))
                    ]
                else:
                    df = self.placements[(self.placements["exec_plan"].isin(exec_plan))]
                break
            if len(df) != 0:
                return round(duration / df["iter_time"].values[0])
        xs = ["num_nodes", "num_gpus"]
        ys = ["iter_time"]
        df = self.placements.groupby(xs)[xs + ys].mean()
        num_nodes, num_replicas = len(placement), sum(placement)
        num_nodes = min(num_nodes, 16)
        interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
        ret = interpolator([num_nodes, num_replicas])[0]
        return ret

    @memoize
    def get_throughput(self, placement, exec_plan, num_cpus=1):
        # Normalize placement to the lexicographically smallest rotation.
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i] for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_gpus"]
        ys = ["overlap_sync", "optimize", "iter_time"]
        exec_plan = self.trans_strategy(exec_plan)
        if placement_id in self.placements.placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[
                (self.placements.placement == placement_id)
                & (self.placements["exec_plan"].isin(exec_plan))
            ]
            if len(df) == 0:
                df = self.placements[
                    (self.placements.placement == placement_id)
                    & (self.placements["exec_plan"].isin(["ga"]))
                ]
            if len(df) != 0:
                ret = df[ys].values[0]
                return ret
                # Interpolate between num_nodes, num_replicas, and local_bsz.
        df = self.placements.groupby(xs)[xs + ys].mean()
        num_nodes, num_replicas = len(placement), sum(placement)
        num_nodes = min(num_nodes, 16)
        interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
        ret = interpolator([num_nodes, num_replicas])[0]
        return ret

    @memoize
    def get_pollux_throughput(self, placement, local_bsz):
        # Normalize placement to the lexicographically smallest rotation.
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i] for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_replicas", "local_bsz"]
        ys = ["step_time", "sync_time"]
        if placement_id in self.placements.placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[self.placements.placement == placement_id]
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0)
            ret = interpolator(local_bsz)
        else:
            # Interpolate between num_nodes, num_replicas, and local_bsz.
            df = self.placements.groupby(xs)[xs + ys].mean()
            df = df.append(self.scalability, ignore_index=True)
            num_nodes, num_replicas = len(placement), sum(placement)
            num_nodes = min(num_nodes, 16)
            interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
            ret = interpolator([num_nodes, num_replicas, local_bsz])[0]
        assert sum(ret) == sum(ret), "{} {} {}".format(self.name, placement, local_bsz)
        return ret

    def trans_strategy(self, exec_plan):
        if type(exec_plan) == tuple:
            if "zero-dp" in exec_plan:
                return ["zero-dp"]
            if "zero-offload" in exec_plan:
                return ["zero-offload"]
        if exec_plan == "3D":
            exec_plan = ["DDP", "MP"]
        else:
            exec_plan = [exec_plan]
        return exec_plan


TRACES_DIR = os.path.join(os.path.dirname(__file__), "../simulator/traces")
print(TRACES_DIR)
APPLICATIONS = {
    "vit": Application(os.path.join(TRACES_DIR, "vit")),
    "roberta": Application(os.path.join(TRACES_DIR, "roberta")),
    "bert": Application(os.path.join(TRACES_DIR, "bert")),
    "t5": Application(os.path.join(TRACES_DIR, "t5")),
    "gpt2": Application(os.path.join(TRACES_DIR, "gpt2")),
    "llama7": Application(os.path.join(TRACES_DIR, "llama7")),
    "llama30": Application(os.path.join(TRACES_DIR, "llama30")),
}
