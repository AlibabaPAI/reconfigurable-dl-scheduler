import csv


class SimpleScheduler:
    def __init__(self, num_gpus=0):
        self.num_gpus = num_gpus
        self.num_jobs = 0

        self.allocation = {}
        self.jobs = []

    def submit(self, job_name):
        self.jobs.append(job_name)
        self.num_jobs += 1

    def allocate(self):
        gpu_per_job = self.num_gpus // self.num_jobs
        for job in self.jobs:
            self.allocation[job] = gpu_per_job
        self.allocation[job] += self.num_gpus % self.num_jobs

    def simulate(self):
        job_performance = {}

        for job in self.jobs:
            job_trace = "../../simulator/traces/" + job + "/placements.csv"
            job_performance.setdefault(job, [100.0, 100.0])
            with open(job_trace, "r", encoding="utf-8-sig") as data:
                reader_config = csv.DictReader(data)
                data_rows = list(reader_config)
            for data_row in data_rows:
                if data_row["placement"] == str(self.allocation[job]):
                    if job_performance[job][0] > float(data_row["iter_time"]):
                        job_performance[job][0] = float(data_row["iter_time"])
                        job_performance[job][1] = data_row["exec_plan"]
        return job_performance
