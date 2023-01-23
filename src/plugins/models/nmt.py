from typing import Dict

from plugins import PluginBase
from plugins.models.base import ModelBase
from plugins.models.config import NMTConfig


class NMTModel(ModelBase):
    """
    NMT Model class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        user_override = kwargs["model_config"] if "model_config" in kwargs and kwargs["model_config"] else {}
        self.config = NMTConfig(user_override)

    def populate_payload(self, payload: Dict, raw_data: list, source_language: str, target_language: str, **kwargs):
        payload["text"] = raw_data[0]
        payload["source_language"] = source_language[0]
        payload["target_language"] = target_language[0]
        # print("payload: ", payload)
        return payload
