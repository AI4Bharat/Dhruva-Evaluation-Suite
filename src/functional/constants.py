from enum import Enum
from typing import Any, Optional, Union
from pathlib import Path
import yaml
from pydantic import BaseSettings as PydanticBaseSettings, BaseModel, Extra, FilePath
from pydantic.env_settings import SettingsSourceCallable
from pydantic.utils import deep_update

from dhruva_logger import logger


class _EnvSettings(PydanticBaseSettings):
    """Initialises all local settings from the env"""
    api_key: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        fields = {
            'api_key': {
                'env': 'DHRUVA_API_KEY',
            },
        }


# Enums
class _TasksEnum(BaseModel):
    NMT: str = "nmt"
    TTS: str = "tts"
    ASR: str = "asr"
    NER: str = "ner"

class _ModelTypesEnum(BaseModel):
    REST: str = "rest"
    STREAMING: str = "streaming"

class _DatasetsEnum(BaseModel):
    FLORES: str = "facebook/flores"
    MUCS: str = "ai4bharat/MUCS-internal"

class _Enums(BaseModel):
    tasks: _TasksEnum = _TasksEnum()
    model_type: _ModelTypesEnum = _ModelTypesEnum()
    datasets: _DatasetsEnum = _DatasetsEnum()
    # tasks: Enum = Enum("task", "NMT TTS ASR NER")
    # model_type: Enum = Enum("model_type", "REST STREAMING")
    # datasets: Enum = Enum("datasets", "FLORES")

Enums = _Enums()


# User config
class _Task(BaseModel):
    name: str
    type: str
    metric: str

class _Model(BaseModel):
    type: str
    url: str

class _Dataset(BaseModel):
    name: str
    path: str
    source_language: Optional[Union[list[str], str]]
    target_language: Optional[Union[list[str], str]]
    split: str
    # subset: ""
    # class Config:
    #     extra = Extra.allow

class UserConfiguration(PydanticBaseSettings):
    """Initialises all user configs from a yaml file"""

    model: _Model
    task: _Task
    dataset: Union[list[_Dataset], _Dataset]

    class Config:
        config_files = [Path("test-nmt.yml")]

        # @classmethod
        # def customise_sources(
        #         cls,
        #         init_settings: SettingsSourceCallable,
        #         env_settings: SettingsSourceCallable,
        #         file_secret_settings: SettingsSourceCallable
        # ) -> tuple[SettingsSourceCallable, ...]:
        #     return init_settings, env_settings, config_file_settings


# def config_file_settings(settings: PydanticBaseSettings) -> dict[str, Any]:
#     config: dict[str, Any] = {}
#     if not isinstance(settings, PydanticBaseSettings):
#         return config
#     for path in settings.Config.config_files:
#         if not path.is_file():
#             logger.error(f"No file found at `{path.resolve()}`")
#             continue
#         logger.info(f"Reading config file `{path.resolve()}`")
#         if path.suffix in {".yaml", ".yml"}:
#             config = deep_update(config, parse_yaml_file(path))
#         else:
#             logger.error(f"Unknown config file extension `{path.suffix}`")
#     return config


# def parse_yaml_file(yaml_file_path):
#     """ Load the YAML file contents into a dictionary"""
#     with open(yaml_file_path, "r") as f:
#         yaml_data = yaml.safe_load(f)
#         if not isinstance(yaml_data, dict):
#             raise TypeError(
#                 f"Config file not found: {yaml_file_path}"
#             )
#     return yaml_data


# Lookups
ULCA_LANGUAGE_CODE_TO_FLORES_MAPPING = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ml": "mal_Mlym",
    "sn": "san_Deva",
    "pa": "pan_Guru",
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "or": "ory_Orya",
    "ur": "urd_Arab",
    "ka": "kan_Knda",
    "gu": "guj_Gujr",
}

FLORES_TO_ULCA_LANGUAGE_CODE_MAPPING = {
    val: key for key, val in ULCA_LANGUAGE_CODE_TO_FLORES_MAPPING.items()
}

DATASET_INPUT_COLUMN_MAPPING = {
    "facebook/flores": "sentence_"
}

EnvSettings = _EnvSettings()
