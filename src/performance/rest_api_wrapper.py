import glob
import time
import datetime
import subprocess
import yaml
import pandas as pd
from constants import UserConfiguration
import argparse

parser = argparse.ArgumentParser(description="Process four input strings")


def parse_yaml_file(yaml_file_path):
    """Load the YAML file contents into a dictionary"""
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
        if not isinstance(yaml_data, dict):
            raise TypeError(f"Config file not found: {yaml_file_path}")
    return yaml_data


# Add the arguments
parser.add_argument("-f", "--file", default="")

args = parser.parse_args()
user_config = UserConfiguration.parse_obj(parse_yaml_file(args.file))

test_params = user_config.params.test_params


def normalise_latencies(lat):
    unit = lat[-1]
    if unit == "s":
        return int(lat[:-1])
    elif unit == "m":
        return int(lat[:-1]) * 60
    elif unit == "h":
        return int(lat[:-1]) * 3600


def run_tests(task):
    for pno, params in test_params.items():
        print(params)
        now = datetime.datetime.now()
        filename = f"results/pno_{pno}_payload_{user_config.params.payload_path.split('_')[-2]}_{user_config.params.payload_path.split('_')[-1][:-4]}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.txt"

        print(user_config.model.url)
        command = [
            "wrk2",
            f"-t{params['threads']}",
            f"-c{params['connections']}",
            f"-d{params['duration']}",
            f"-R{params['rps']}",
            "-s",
            f"{user_config.params.payload_path}",
            "--latency",
            f"{user_config.model.url}",
        ]
        status = subprocess.run(command, shell=False, capture_output=True)
        print("status: ", status.stdout.decode("utf-8"))
        with open(filename, "w") as o:
            o.write(f"Requested RPS : {params['rps']}\n")
            o.write(status.stdout.decode("utf-8"))

        print("--------------------")
        time.sleep(1)


if __name__ == "__main__":
    run_tests("asr")
