import json
import base64
import logging
import requests
import itertools
from typing import List
import multiprocessing as mp

from tqdm import tqdm
from config.schema.services.request import (
    ULCAAsrInferenceRequest,
    ULCATtsInferenceRequest,
    ULCANerInferenceRequest,
    ULCATranslationInferenceRequest
)

from config.schema.services.response import (
    ULCAAsrInferenceResponse,
    ULCATtsInferenceResponse,
    ULCANerInferenceResponse,
    ULCATranslationInferenceResponse
)
# from transformers.pipeline import Pipeline

BATCH_LEN = 5
ASR_TASK = "dhruva_asr"



ULCATaskRequestSchemaMapping = {
    "dhruva_asr": ULCAAsrInferenceRequest,
    "dhruva_tts": ULCATtsInferenceRequest,
    "dhruva_ner": ULCANerInferenceRequest,
    "dhruva_nmt": ULCATranslationInferenceRequest,
}

ULCATaskResponseSchemaMapping = {
    "dhruva_asr": ULCAAsrInferenceResponse,
    "dhruva_tts": ULCATtsInferenceResponse,
    "dhruva_ner": ULCANerInferenceResponse,
    "dhruva_nmt": ULCATranslationInferenceResponse,
}


def _encode_audio(raw_input):
    # return base64.b64encode(raw_input).decode("utf-8")
    return raw_input


def generate_asr_payload(batch_data, input_column, language_column):
    payload = ULCATaskRequestSchemaMapping.get(ASR_TASK)
    payload.config.language.sourceLanguage = batch_data[0][language_column]
    payload.audio = [{"audioContent": _encode_audio(data[input_column])} for data in batch_data]
    return payload


def parse_asr_response(response):
    payload = ULCATaskResponseSchemaMapping.get(ASR_TASK)(response)
    return [{"text": p.source} for p in payload.output]


class DhruvaModel():
    def __init__(self, task: str, url: str, input_language_column: str, input_column: str, api_key: str, output_language_column: str = "", **kwargs):
        self.task = task
        self.url = url
        self.input_language_column = input_language_column
        self.input_column = input_column
        self.headers = {"Authorization": api_key}

    def _infer(self, batch_data: List):
        self.payload = None
        if self.task == ASR_TASK:
            self.payload = generate_asr_payload(batch_data, self.input_column, self.input_language_column)

        if self.payload is None:
            raise ValueError("Empty payload")

        results = requests.post(self.url, data=json.dumps(self.payload), headers=self.headers).json()

        parsed_results = None
        if self.task == ASR_TASK:
            parsed_results = parse_asr_response(results)

        return parsed_results

    def infer_batch(self, all_data):
        all_results = []
        batch_data = []

        # .cache/huggingface/
        # Later, write all outputs to some file
        for data in tqdm(all_data):
            batch_data.append(data)
            if len(batch_data) == BATCH_LEN:
                all_results.extend(self._infer(batch_data))
                batch_data = []

        if batch_data:
            all_results.extend(self._infer(batch_data))
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
                ]
            )
            all_results = list(itertools.chain(*results))
        return all_results
