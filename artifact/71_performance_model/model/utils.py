import csv
from pred_throughput import PerfParams


def load_profile_from_files(config_file, data_file):
    profile = {}

    with open(config_file, "r", encoding="utf-8-sig") as f_config:
        reader_config = csv.DictReader(f_config)
        config_rows = list(reader_config)

    with open(data_file, "r", encoding="utf-8-sig") as f_data:
        reader_data = csv.DictReader(f_data)
        data_rows = list(reader_data)

    for config_row, data_row in zip(config_rows, data_rows):
        key = (
            int(config_row["nodes"]),
            int(config_row["gpus"]),
            float(config_row["parameter_size"]),
            int(config_row["batch_size"]),
            int(config_row["seq_hid"]),
            int(config_row["num_cpu_threads"]),
            int(config_row["zero_type"]),
            int(config_row["is_tp"]),
            int(config_row["is_pp"]),
        )

        forward_time = int(data_row["forward_time"])
        Tcc = int(data_row["Tcc"])
        Topt = int(data_row["Topt"])
        Titer = int(data_row["iter_time"])

        strategy = [int(x) for x in data_row["strategy"]]

        value = (forward_time, Tcc, Topt, Titer, strategy)

        profile[key] = value

    return profile


def test_load_function():
    config_file = "./artifact/performance_model/model/profile/vit/config.csv"
    data_file = "./artifact/performance_model/model/profile/vit/data.csv"

    profile = load_profile_from_files(config_file, data_file)

    for key, value in profile.items():
        print(f"profile{key} = {value}")


def load_perf_params_from_csv(file_path, target_model):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"].strip().lower() == target_model.strip().lower():
                params = PerfParams(
                    k_bwd=float(row["k_bwd"]),
                    k_sync=float(row["k_sync"]),
                    k_opt=float(row["k_opt"]),
                    k_os_offload=float(row["k_os_offload"]),
                    k_const=float(row["k_const"]),
                    k_off=float(row["k_off"]),
                    k_swap=float(row["k_swap"]),
                )
                return params

        raise ValueError(f"Model '{target_model}' not found in the CSV file.")
