from typing import List
from dataclasses import dataclass, field


@dataclass
class BaseConfig():
    MODEL_NAME: str
    MODEL_VERSION: str

    # BASE_URL: str
    METRICS_URL: str = ""
    HTTP_URL: str = ""
    GRPC_URL: str = ""

    # Needed for bare Triton instance
    # HTTP_PORT: str = "8001"
    # GRPC_PORT: str = "8001"
    # METRICS_PORT: str = "8002"

    # def generate_url(self, url, port):
    #     return url + "/" + port

    # def __post_init__(self):
    #     if not self.HTTP_URL:
    #         self.HTTP_URL = self.generate_url(self.BASE_URL, self.HTTP_PORT)

    #     if not self.GRPC_URL:
    #         self.GRPC_URL = self.generate_url(self.BASE_URL, self.GRPC_PORT)

    #     if not self.METRICS_URL:
    #         self.METRICS_URL = self.generate_url(self.BASE_URL, self.METRICS_PORT)


@dataclass
class ASRBatchConfig(BaseConfig):
    MODEL_NAME: str = "e2e"
    MODEL_VERSION: str = "1"

    # Network
    HTTP_URL: str = "aml-asr-hi-endpoint.eastus.inference.ml.azure.com"
    API_KEY: str = "9i2vidTyIdmWO1vpDbFJAk8trK2J5rTS"


@dataclass
class IndicTinyASRConfig(BaseConfig):
    MODEL_NAME: str = "IndicTinyASR"
    MODEL_VERSION: str = "1"

    # Network
    HTTP_URL: str = "https://asr-api.ai4bharat.org/asr/v1/recognize/hi"
    API_KEY: str = "9i2vidTyIdmWO1vpDbFJAk8trK2J5rTS"
    language_code = "hi"
    sampling_rate = 48000
    post_processors = []