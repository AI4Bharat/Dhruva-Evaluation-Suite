import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class BaseConfig():
    MODEL_NAME: str
    MODEL_VERSION: str

    HTTP_DOMAIN: str
    HTTP_URL: str
    HTTP_PROTOCOL: str = "http://"
    # HTTP_URL: str = "aml-asr-hi-endpoint.eastus.inference.ml.azure.com"

    API_KEY: str = os.getenv("DHRUVA_API_KEY")


@dataclass
class ASRBatchConfig(BaseConfig):
    MODEL_NAME: str = "offline_conformer"
    MODEL_VERSION: str = "1"
    MODEL_TYPE: str = "ASR"
    HTTP_PROTOCOL: str = "http://"
    HTTP_DOMAIN: str = "api.dhruva.ai4bharat.org:8090"  # "127.0.0.1:8001"
    # HTTP_DOMAIN: str = "host.docker.internal:8001"
    HTTP_URL: str = "/infer"


@dataclass
class NMTConfig(BaseConfig):
    MODEL_NAME: str = "nmt"
    MODEL_VERSION: str = "1"
    MODEL_TYPE: str = "NMT"
    HTTP_PROTOCOL: str = "http://"
    HTTP_DOMAIN: str = "127.0.0.1:8001"
    # HTTP_DOMAIN: str = "host.docker.internal:8001"
    HTTP_URL: str = "/infer_nmt"


headers = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://models.ai4bharat.org",
    # "Content-Length": "147503",  # seems this is not needed to be calculated every time
    "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
    "Host": "asr-api.ai4bharat.org",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Referer": "https://models.ai4bharat.org/",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

ULCAFormats = {
    "ASR": {
        "headers": headers,
        "payload": {
            "config": {
                "language": {
                    "sourceLanguage": "hi",
                },
                "transcriptionFormat": {
                    "value": "transcript",
                },
                "audioFormat": "wav",
                "samplingRate": "16000",
                "postProcessors": None
            },
            "audio": [{"audioContent": ""}],
        }
    },
    "NMT": {
        "headers": headers,
        "payload": { "text": "", "source_language": "", "target_language": "" }
    }
}
