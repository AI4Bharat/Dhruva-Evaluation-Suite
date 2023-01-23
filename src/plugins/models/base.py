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
        # For overriding
        return raw_data

    def populate_payload(self, payload: Dict, raw_data: List, **kwargs):
        raise NotImplementedError

    def generate_ULCA_payload(self, raw_data, model_type, **kwargs):
        headers, payload = ULCAFormats[model_type]["headers"], ULCAFormats[model_type]["payload"]
        payload = self.populate_payload(payload, raw_data, **kwargs)
        return headers, payload

    def invoke(self, *args, **kwargs):
        raw_data = self.get_inputs(kwargs["input"])
        headers, payload = self.generate_ULCA_payload(raw_data, self.config.MODEL_TYPE, **kwargs)
        return self.infer(self.config.HTTP_PROTOCOL + self.config.HTTP_DOMAIN + self.config.HTTP_URL, headers, payload, kwargs.get("client"))

    def infer(self, url: str, headers: Dict, payload: Dict, client=None):
        # self._logger.debug("inside infer")
        # For overriding locust client for performance testing
        if not client:
            client = requests

        # self._logger.debug("client")

        # self._logger.debug(f"url: {url}")
        # self._logger.debug(requests.post(url, headers=headers, data=json.dumps(payload)))
        # self._logger.debug(requests.post(url, headers=headers, data=json.dumps(payload)).json())
        with open("test_payload", "w") as f:
            f.write("payload=" + json.dumps(payload))
            f.write("headers=" + json.dumps(headers))

        # self._logger.debug(requests.post(url, headers=headers, data=json.dumps(payload)))
        return client.post(url, headers=headers, data=json.dumps(payload), timeout=5).json()
        # return {"data": {"key": "value"}}
        # return client.get("http://api.dhruva.ai4bharat.org:8090/")
        # print("url: ", url)
        # print(client)
        # print(client.post("http://127.0.0.1:8000" + url, headers=headers, data=json.dumps(payload)))
