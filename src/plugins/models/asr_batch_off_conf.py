from typing import Dict

from plugins import PluginBase
from plugins.models.base import ModelBase
from plugins.models.config import ASRBatchConfig


class ASRBatchOffConfModel(ModelBase):
    """
    ASR Model class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        user_override = kwargs["model_config"] if "model_config" in kwargs and kwargs["model_config"] else {}
        self.config = ASRBatchConfig(user_override)

    def populate_payload(self, payload: Dict, raw_data: list):
        payload["audio"][0]["audioContent"] = raw_data
        return payload
