import json
import requests
from typing import Dict, List

import numpy as np

from plugins import PluginBase
from plugins.models.config import ASRBatchConfig, ULCAFormats


class ModelBase(PluginBase):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        # self.config = kwargs["model_config"] if "model_config" in kwargs else None
        self.http_client = requests
        if "http_client" in kwargs and kwargs["http_client"]:
            self.client = kwargs["http_client"]

    def get_inputs(self, raw_data: list):
        return raw_data

    def populate_payload(self, payload: Dict, raw_data: List):
        raise NotImplementedError

    def generate_ULCA_payload(self, raw_data, model_type: str="ASR"):
        headers, payload = ULCAFormats[model_type]["headers"], ULCAFormats[model_type]["payload"]
        payload = self.populate_payload(payload, raw_data)
        return headers, payload

    def invoke(self, *args, **kwargs):

        ''' ASR models require the input to be a List '''
        if(kwargs["model_type"] == "ASR"):
            raw_data = self.get_inputs(kwargs["input"])


        ''' TTS models require the input to be a String'''
        if(kwargs["model_type"] == "TTS"):
            raw_data = str(self.get_inputs(kwargs["input"]))
        

        headers, payload = self.generate_ULCA_payload(raw_data, self.config.MODEL_TYPE)
        return self.infer(self.config.HTTP_PROTOCOL + self.config.HTTP_DOMAIN + self.config.HTTP_URL, headers, payload)

    def infer(self, url: str, headers: Dict, payload: Dict, client=None):
        # For overriding locust client for performance testing
        if not client:
            client = self.http_client
        return client.post(url, headers=headers, data=json.dumps(payload)).json()
