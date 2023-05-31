import json
import os
import glob
import time
import datetime
import subprocess

import pandas as pd

import argparse
import base64
parser = argparse.ArgumentParser(description='Process four input strings')

# Add the arguments
parser.add_argument('--task', help='ASR, NMT OR TTS', required=True)
parser.add_argument('--lua_file', help='Name of the lua file', required=True)
parser.add_argument('--url', help='URL of the API endpoint', required=True)
parser.add_argument('--result_folder_name', help='Location ID for the results files', required=True)


args = parser.parse_args()

test_params = {
    1: {"threads": 30, "connections": 140, "rps": 55, "duration": "30s"},
    # 2: {"threads": 5, "connections": 10, "rps": 2, "duration": "2m"},
    # 3: {"threads": 5, "connections": 10, "rps": 4, "duration": "2m"},
    # 4: {"threads": 15, "connections": 80, "rps": 70, "duration": "2m"},
    # 5: {"threads": 15, "connections": 100, "rps": 80, "duration": "2m"},
    # 3: {"threads": 5, "connections": 20, "rps": 4, "duration": "2m"},
    # 4: {"threads": 5, "connections": 10, "rps": 2, "duration": "30s"},
    # 5: {"threads": 5, "connections": 10, "rps": 2, "duration": "2m"},
    # 6: {"threads": 5, "connections": 10, "rps": 4, "duration": "2m"},
    # 4: {"threads": 5, "connections": 10, "rps": 8, "duration": "2m"},
    # 5: {"threads": 5, "connections": 15, "rps": 6, "duration": "2m"},
    # 6: {"threads": 10, "connections": 40, "rps": 24, "duration": "2m"},
    # 7: {"threads": 10, "connections": 80, "rps": 40, "duration": "2m"},
    # 8: {"threads": 10, "connections": 50, "rps": 33, "duration": "2m"},
    # 6: {"threads": 20, "connections": 80, "rps": 40, "duration": "25m"},
    # 7: {"threads": 30, "connections": 100, "rps": 52, "duration": "25m"},
    # 8: {"threads": 40, "connections": 140, "rps": 64, "duration": "2m"},
    # 9: {"threads": 3, "connections": 10, "rps": 3, "duration": "2m"},
    # 7: {"threads": 60, "connections": 160, "rps": 130, "duration": "2m"},
    # 8: {"threads": 70, "connections": 180, "rps": 140, "duration": "2m"},
    # 6: {"threads": 15, "connections": 70, "rps": 10, "duration": "2m"},
    # 7: {"threads": 10, "connections": 60, "rps": 20, "duration": "2m"},
    # 8: {"threads": 15, "connections": 80, "rps": 25, "duration": "2m"},
    # 4: {"threads": 20, "connections": 80, "rps": 40, "duration": "2m"},
    # 8: {"threads": 25, "connections": 100, "rps": 51, "duration": "2m"},
    # 8: {"threads": 10, "connections": 100, "rps": 20, "duration": "2m"},
    # 9: {"threads": 20, "connections": 100, "rps": 50, "duration": "10m"},
    # 9: {"threads": 40, "connections": 160, "rps": 80, "duration": "2m"},
    # 10: {"threads": 50, "connections": 200, "rps": 100, "duration": "2m"},
    # 11: {"threads": 30, "connections": 170, "rps": 150, "duration": "5m"},
    #10: {"threads": 40, "connections": 150, "rps": 120, "duration": "5m"},
    #11: {"threads": 40, "connections": 180, "rps": 150, "duration": "2m"},
    #12: {"threads": 50, "connections": 200, "rps": 180, "duration": "2m"},,
    # 12: {"threads": 50, "connections": 200, "rps": 62, "duration": "2m"},
    # 13: {"threads": 60, "connections": 250, "rps": 125, "duration": "2m"},
    # 14: {"threads": 700, "connections": 300, "rps": 150, "duration": "2m"},
    # 15: {"threads": 100, "connections": 400, "rps": 200, "duration": "2m"}
}


def parse_file(path):    
    plot_name = test_params[path.split("_")[2]]
    results = [plot_name]
    columns = ["plot_name"]

    with open(path, "r") as f:
        for line in f.read():
            if not flag:
                if not "Latency Distribution (HdrHistogram - Recorded Latency)" in line:
                    continue

            flag = 1
            if flag > 1:
                percentile, latency = line.split("\t")
                percentile = percentile.strip()[:-1]
                latency = latency.strip()
                latency = normalise_latencies(latency)
                results.append(latency)
                columns.append(percentile)

            flag += 1
            if flag > 9:
                break
                flag = 0
    return pd.DataFrame([results], columns=columns)


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
            #f"http://api.dhruva.ai4bharat.org:8090/infer_{task}_hi",
            # f"https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat%2Fconformer-multilingual-dravidian-gpu--t4"
            # "https://asr-api.ai4bharat.org/asr/v1/recognize/en",
            #"https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat%2Fconformer-multilingual-indo_aryan-gpu--t4",
            # "https://api.dhruva.ai4bharat.org/services/inference/asr_e2e?serviceId=ai4bharat%2Fconformer-hi-gpu--t4",
            #"https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat%2Fconformer-en-gpu--t4",
            #"http://api.dhruva.ai4bharat.org:8090/infer_asr_en",
            # "|",
            # "tee",
            # filename
            f"{args.url}",
        ]
        print(command)
        # subprocess.run(["ls", "-l", "/dev/null"])
        # command = f"wrk2 -t{params['threads']} -c{params['connections']} -d2m -R{params['rps']} -s nmt.lua --latency http://api.dhruva.ai4bharat.org:8090/infer_nmt | tee ./results/nmt_histo_{pno}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.txt"
        status = subprocess.run(command, shell=False, capture_output=True)
        print("status: ", status.stdout.decode("utf-8"))
        with open(filename, "w") as o:
            o.write(f"Requested RPS : {params['rps']}\n")
            o.write(status.stdout.decode("utf-8"))

        print("--------------------")
        time.sleep(1)


if __name__ == "__main__":
    run_tests("asr")
    # parse_files()



#Example Command
#python3 load_test_helper.py --task "ASR" --lua_file "ASR_8Sec.lua" --url "https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat%2Fconformer-en-gpu--t4" --result_folder_name "results_vad"
#python3 load_test_helper.py --task "TTS" --lua_file "TTS_test.lua" --url "https://api.dhruva.ai4bharat.org/services/inference/tts?serviceId=ai4bharat%2Findic-tts-coqui-misc-gpu--t4" --result_folder_name "results9"
#python3 load_test_helper.py --task "NMT" --lua_file "NMT_Hindi_10.lua" --url "https://api.dhruva.ai4bharat.org/services/inference/nmt?serviceId=ai4bharat%2Findictrans-fairseq-all-gpu--t4" --result_folder_name "results9"
