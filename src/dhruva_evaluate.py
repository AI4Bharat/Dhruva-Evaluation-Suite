import os
import json
import tarfile
import logging
from pathlib import Path
from typing import List, Any, Dict, Tuple

from datasets import Dataset
from tqdm import tqdm
import numpy as np
import soundfile as sf

# import helpers
# from plugins.preprocessors.speech_transcript_cleaning import cleaning_pipeline, get_dict_chars


ULCAFormats = {
    "ASR": {
        # "headers": headers,
        "payload": {
            "audio": [
                {
                "audioContent": "string",
                "audioUri": "string"
                }
            ],
            "config": {
                "language": {
                    "sourceLanguage": "string"
                    },
                "audioFormat": "wav",
                "encoding": "base64",
                "samplingRate": 16000,
                "postProcessors": [None]
            }
        }
    },
    "NMT": {
        # "headers": headers,
        "payload": { "text": "", "source_language": "", "target_language": "" }
    }
}




# class DhruvaModel():
#     def __init__(self, task: str, lang: str, **kwargs):
#         self.task = task
#         self.url =  "https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat/conformer-hi-gpu--t4"
#         self.payload = ULCAFormats.get(self.task)["payload"]
#         self.payload["config"]["language"]["sourceLanguage"] = lang

#     def __call__(self, batch_audio, **kwargs):
#         import requests
#         self.payload["audio"] = [{"audioContent": audio} for audio in batch_audio]
#         results = requests.post(self.url, payload=json.dumps(self.payload)).json()
#         return [{"generated_transcript": p["source"]} for p in results["output"]]



if __name__ == "__main__":
    from datasets import load_dataset, Audio, Dataset
    from dhruva_models import DhruvaModel
    from dhruva_evaluators import DhruvaASREvaluator
    audio_path = "/Users/ashwin/ai4b/perf_testing/Dhruva-Evaluation-Suite/datasets/raw/MUCS/Hindi"
    dataset = load_dataset("./dhruva_datasets/asr.py", "MUCS-hi", data_dir=audio_path, split="test")  # , streaming=True
    print(dataset.column_names, dataset.shape, dir(dataset), isinstance(dataset, Dataset))
    # print(dataset["audio"])
    dataset = dataset[:10]

    from evaluate import evaluator
    from datasets import load_dataset
    task_evaluator = evaluator("automatic-speech-recognition")
    url = "https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat/conformer-hi-gpu--t4"
    results = task_evaluator.compute(
        model_or_pipeline=DhruvaModel(url=url, task="automatic-speech-recognition", lang="hi"),
        data="./dhruva_datasets/asr.py",
        subset="MUCS-hi",
        split="test",
        input_column="audio",
        label_column="transcript",
        metric="wer",
    )
    print("----> ", results)


# Load using custom script
# Structure:
# {
#     "audio":{"path": "", "array": [], "sampling_rate": 16000},
#     "client_id": 1,
#     "locale": "hi",
#     "age": 40,
#     "accent": "",
#     "gender": "",
#     "segment": "",
#     "transcript": ""
# }

# Hit Dhruva
# Gen WER

# NMT
# Structure

# Hit Dhruva
# Gen BLEU