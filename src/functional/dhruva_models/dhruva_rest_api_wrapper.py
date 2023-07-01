import json
import base64
import logging
import requests
import itertools
from typing import List
import multiprocessing as mp

import datasets
from tqdm import tqdm
from constants import Enums
from schema.services.request.ulca_asr_inference_request import ULCAAsrInferenceRequest
from schema.services.request.ulca_tts_inference_request import ULCATtsInferenceRequest
from schema.services.request.ulca_ner_inference_request import ULCANerInferenceRequest
from schema.services.request.ulca_translation_inference_request import (
    ULCATranslationInferenceRequest,
)


from schema.services.response.ulca_asr_inference_response import (
    ULCAAsrInferenceResponse,
)
from schema.services.response.ulca_tts_inference_response import (
    ULCATtsInferenceResponse,
)
from schema.services.response.ulca_ner_inference_response import (
    ULCANerInferenceResponse,
)
from schema.services.response.ulca_translation_inference_response import (
    ULCATranslationInferenceResponse,
)

BATCH_LEN = 5


ULCATaskRequestSchemaMapping = {
    Enums.tasks.ASR: ULCAAsrInferenceRequest,
    Enums.tasks.TTS: ULCATtsInferenceRequest,
    Enums.tasks.NER: ULCANerInferenceRequest,
    Enums.tasks.NMT: ULCATranslationInferenceRequest,
}

ULCATaskResponseSchemaMapping = {
    Enums.tasks.ASR: ULCAAsrInferenceResponse,
    Enums.tasks.TTS: ULCATtsInferenceResponse,
    Enums.tasks.NER: ULCANerInferenceResponse,
    Enums.tasks.NMT: ULCATranslationInferenceResponse,
}

feature = datasets.Audio()

def _encode_audio(raw_input):
    data = feature.encode_example(raw_input)
    return base64.b64encode(data["bytes"]).decode("utf-8")


def generate_asr_payload(batch_data: list, input_column: str, language_column: str):
    payload = {
        "config": {
            "language": {},
            "audioFormat": "wav",
            "samplingRate": 16000,
            "postProcessors": [],
        }
    }
    payload["config"]["language"]["sourceLanguage"] = batch_data[0][language_column]
    payload["audio"] = [
        {"audioContent": _encode_audio(data[input_column])}
        for data in batch_data
    ]
    payload = ULCAAsrInferenceRequest(**payload)
    return payload.dict()


def parse_asr_response(response: dict):
    payload = ULCAAsrInferenceResponse(**response)
    return [{"text": p.source} for p in payload.output]


def generate_nmt_payload(batch_data: list, input_column: str):
    payload = {
        "config": {
            "language": {
                "sourceLanguage": batch_data[0]["source_language"],
                "sourceScriptCode": "",
                "targetLanguage": batch_data[0]["target_language"],
                "targetScriptCode": ""
            },
            "postProcessors": [],
        }
    }
    payload["input"] = [{"source": data[input_column]} for data in batch_data]
    payload = ULCATranslationInferenceRequest(**payload)
    return payload.dict()


def parse_nmt_response(response: dict):
    payload = ULCATranslationInferenceResponse(**response)
    return [{"text": p.target} for p in payload.output]


class DhruvaRESTModel:
    def __init__(
        self,
        task: str,
        url: str,
        input_column: str,
        api_key: str,
        **kwargs
    ):
        self.task = task
        self.url = url
        self.service_id = None
        self.input_language_column = kwargs.get("input_language_column")
        self.input_column = input_column
        self.headers = {"Authorization": api_key}

    def _infer(self, batch_data: List):
        try:
            self.payload = globals()[f"generate_{self.task}_payload"](batch_data, self.input_column)
            if self.payload is None:
                raise ValueError("Empty payload")

            results = requests.post(
                self.url, data=json.dumps(self.payload), headers=self.headers, timeout=30
            ).json()

            parsed_results = globals()[f"parse_{self.task}_response"](results)

        except Exception as e:
            import traceback
            print(traceback.print_exc(e))

        return parsed_results

    def infer_batch(self, all_data):
        all_results = []
        batch_data = []

        # .cache/huggingface/
        # Later, write all outputs to some file
        for data in tqdm(all_data):
            # print(data)
            batch_data.append(data)
            if len(batch_data) == BATCH_LEN:
                results = self._infer(batch_data)
                if results is not None:
                    all_results.extend(results)
                    batch_data = []
                else :
                    print(data["path"])

        if batch_data:
            results = self._infer(batch_data)
            all_results.extend(results)
        return all_results

    def __call__(self, all_audios, **kwargs):
        # If we do multiprocesing here, read from generator / list and split between processes
        # Write to pyarrow files from each process
        # collate responses from all files and stream / return

        num_processes = mp.cpu_count()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(
                self.infer_batch,
                [
                    all_audios.shard(num_shards=num_processes, index=i, contiguous=True)
                    for i in range(num_processes)
                ],
            )
            all_results = list(itertools.chain(*results))
        return all_results
