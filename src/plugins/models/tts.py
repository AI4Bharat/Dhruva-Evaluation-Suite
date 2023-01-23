from typing import Dict

from plugins import PluginBase
from plugins.models.base import ModelBase
from plugins.models.config import IndicTTSConfig


class IndicTTSModel(ModelBase):
    """
    TTS Model class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        user_override = kwargs["model_config"] if "model_config" in kwargs and kwargs["model_config"] else {}
        self.config = IndicTTSConfig(user_override)

    def populate_payload(self, payload: Dict, text: list):
        payload["input"][0]["source"] = text
        return payload
