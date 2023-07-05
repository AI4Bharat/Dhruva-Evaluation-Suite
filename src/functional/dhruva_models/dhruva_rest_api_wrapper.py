import json
import base64
import logging
import requests
import itertools
from typing import List
import multiprocessing as mp

import datasets
from tqdm import tqdm
from constants import (
    Enums,
    ULCA_LANGUAGE_CODE_TO_AKSHARANTAR_MAPPING,
    AKSHARANTAR_TO_ULCA_LANGUAGE_CODE_MAPPING,
)
import sys

sys.path.append("..")
from schema.services.request.ulca_asr_inference_request import ULCAAsrInferenceRequest
from schema.services.request.ulca_tts_inference_request import ULCATtsInferenceRequest
from schema.services.request.ulca_ner_inference_request import ULCANerInferenceRequest
from schema.services.request.ulca_translation_inference_request import (
    ULCATranslationInferenceRequest,
)
from schema.services.request.ulca_transliteration_inference_request import (
    ULCATransliterationInferenceRequest,
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
from schema.services.response.ulca_transliteration_inference_response import (
    ULCATransliterationInferenceResponse,
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
    # print(data)
    return base64.b64encode(open(data["path"], "rb").read()).decode("utf-8")


def generate_asr_payload(batch_data: list, input_column: str):
    payload = {
        "controlConfig": {"dataTracking": True},
        "config": {
            "language": {},
            "audioFormat": "wav",
            "samplingRate": 16000,
            "postProcessors": [],
            "encoding": "base64",
        },
    }
    # print(batch_data)
    payload["config"]["language"]["sourceLanguage"] = batch_data[0]["language"]
    if payload["config"]["language"]["sourceLanguage"] == "pa-IN":
        payload["config"]["language"]["sourceLanguage"] = "pa"
    payload["audio"] = [
        {"audioContent": _encode_audio(data["audio"][input_column])}
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
                "targetScriptCode": "",
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


def generate_tts_payload(batch_data, input_column, language_column):
    # For TTS, Batch Length should be taken as 1 to generate the results in the same gender
    payload = {
        "config": {
            "language": {"sourceLanguage": batch_data[0][language_column]},
            "gender": batch_data[0]["gender"],
        },
        "input": {"source": str(batch_data[0][input_column])},
    }
    payload = ULCATtsInferenceRequest(**payload)
    return payload.dict()


def parse_tts_response(response):
    payload = ULCATtsInferenceResponse(**response)
    return [{"audio": p.audioContent} for p in payload.audio]


def generate_transliteration_payload(batch_data, input_column):
    payload = {
        "config": {
            "serviceId": "",
            "language": {
                "sourceLanguage": "en",
                "targetLanguage": "",
            },
            "isSentence": False,
            "numSuggestions": 1,
        }
    }
    # print(batch_data)
    # payload["config"]["language"]["sourceLanguage"] = batch_data[0]["en"]
    payload["config"]["language"][
        "targetLanguage"
    ] = AKSHARANTAR_TO_ULCA_LANGUAGE_CODE_MAPPING.get(
        batch_data[0]["unique_identifier"][:3]
    )
    payload["input"] = [{"source": str(data[input_column])} for data in batch_data]

    payload["controlConfig"] = {"dataTracking": True}
    payload = ULCATransliterationInferenceRequest(**payload)
    return payload.dict()


def parse_transliteration_response(response):
    try:
        payload = ULCATransliterationInferenceResponse(**response)
        return [{"text": p.target} for p in payload.output]
    except:
        print(response)
        return [{"text": [""]} for p in range(BATCH_LEN)]


class DhruvaRESTModel:
    def __init__(
        self,
        task: str,
        url: str,
        input_column: str,
        api_key: str,
        **kwargs,
    ):
        self.task = task
        self.url = url
        self.service_id = None
        self.input_language_column = kwargs.get("input_language_column")
        self.input_column = input_column
        self.headers = {"Authorization": api_key}

    def _infer(self, batch_data: List):
        try:
            self.payload = globals()[f"generate_{self.task}_payload"](
                batch_data, self.input_column
            )
            # file_path = "data.json"

            # # Write the JSON object to the file
            # with open(file_path, 'w') as file:
            #     json.dump(self.payload, file)

            if self.payload is None:
                raise ValueError("Empty payload")
            results = requests.post(
                self.url,
                data=json.dumps(self.payload),
                headers=self.headers,
                timeout=90,
            ).json()
            # print(results)
            parsed_results = globals()[f"parse_{self.task}_response"](results)

        except Exception as e:
            # print()
            import traceback

            print(traceback.print_exc(e))
            return [{"text": ""} for p in range(BATCH_LEN)]

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
                else:
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
