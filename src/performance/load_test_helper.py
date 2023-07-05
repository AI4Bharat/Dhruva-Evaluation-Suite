import glob
import time
import datetime
import subprocess

import pandas as pd

import argparse

parser = argparse.ArgumentParser(description="Process four input strings")

# Add the arguments
parser.add_argument("--task", help="ASR, NMT OR TTS", required=True)
parser.add_argument("--lua_file", help="Name of the lua file", required=True)
parser.add_argument("--url", help="URL of the API endpoint", required=True)
parser.add_argument(
    "--result_folder_name", help="Location ID for the results files", required=True
)


args = parser.parse_args()

test_params = {
    1: {"threads": 30, "connections": 140, "rps": 55, "duration": "30s"},
}


def normalise_latencies(lat):
    unit = lat[-1]
    if unit == "s":
        return int(lat[:-1])
    elif unit == "m":
        return int(lat[:-1]) * 60
    elif unit == "h":
        return int(lat[:-1]) * 3600


def parse_files():
    data = []
    for g in glob.glob("./results_ta_1/*.txt"):
        data.append(parse_file(g))
    pd.concat(data).to_csv("output.csv")


def run_tests(task):
    for pno, params in test_params.items():
        print(params)
        now = datetime.datetime.now()
        filename = f"./{args.task.lower()}_test/{args.result_folder_name}/pno_{pno}_payload_{args.lua_file.split('_')[-2]}_{args.lua_file.split('_')[-1][:-4]}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.txt"

        print(args.url)
        command = [
            "wrk2",
            f"-t{params['threads']}",
            f"-c{params['connections']}",
            f"-d{params['duration']}",
            f"-R{params['rps']}",
            "-s",
            f"{args.task.lower()}_test/{args.lua_file}",
            "--latency",
            f"{args.url}",
        ]
        print(command)
        status = subprocess.run(command, shell=False, capture_output=True)
        print("status: ", status.stdout.decode("utf-8"))
        with open(filename, "w") as o:
            o.write(f"Requested RPS : {params['rps']}\n")
            o.write(status.stdout.decode("utf-8"))

        print("--------------------")
        time.sleep(1)


if __name__ == "__main__":
    run_tests("asr")
