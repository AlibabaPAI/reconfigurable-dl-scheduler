class JobInfo(object):
    def __init__(self, model_name="", ps=0, forward_time=0, local_bsz=0, rsc_metrix=None, max_throughput_config=None, throughput=0,
                 creation_timestamp=0, steps=0, restarts=0, throughput_fn=None, preemptible=True, is_distributed=False, origin_tpt=0, resources=None, speedup_fn=None,
                 attained_service=0, min_replicas=0, max_replicas=0,
                 lease_time=0, job_gpu_demand=0):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
        """
        self.model_name = model_name
        self.ps = ps
        self.forward_time = forward_time
        self.local_bsz = local_bsz
        self.rsc_metrix = rsc_metrix
        self.max_throughput_config = max_throughput_config
        self.throughput = throughput
        self.creation_timestamp = creation_timestamp
        self.steps = steps
        self.restarts = restarts
        self.preemptible = preemptible
        self.throughput_fn = throughput_fn
        self.is_distributed = is_distributed
        self.origin_tpt = origin_tpt

        self.resources = resources
        self.speedup_fn = speedup_fn
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas

        self.lease_time = lease_time
        self.job_gpu_demand = job_gpu_demand


class NodeInfo(object):
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        self.resources = resources
        self.preemptible = preemptible
