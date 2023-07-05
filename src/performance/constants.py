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
        env_file = ".env"
        env_file_encoding = "utf-8"
        fields = {
            "api_key": {
                "env": "DHRUVA_API_KEY",
            },
        }


# Enums
class _TasksEnum(BaseModel):
    NMT: str = "nmt"
    TTS: str = "tts"
    ASR: str = "asr"
    NER: str = "ner"
    Transliteration: str = "transliteration"


class _ModelTypesEnum(BaseModel):
    REST: str = "rest"
    STREAMING: str = "streaming"


class _DatasetsEnum(BaseModel):
    FLORES: str = "facebook/flores"
    MUCS: str = "ai4bharat/MUCS-internal"
    IndicSUPERB: str = "ai4bharat/IndicSUPERB-internal"
    Aksharantar: str = "ai4bharat/aksharantar"


class _Enums(BaseModel):
    tasks: _TasksEnum = _TasksEnum()
    model_type: _ModelTypesEnum = _ModelTypesEnum()
    datasets: _DatasetsEnum = _DatasetsEnum()


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
    input_column: Optional[Union[list[str], str]]
    label_column: Optional[Union[list[str], str]]
    split: str
    subset: Optional[Union[list[str], str]]


class UserConfiguration(PydanticBaseSettings):
    """Initialises all user configs from a yaml file"""

    model: _Model
    task: _Task
    dataset: Union[list[_Dataset], _Dataset]
    results_folder: str = "./results"

    class Config:
        config_files = [Path("test-nmt.yml")]


# Lookups
ULCA_LANGUAGE_CODE_TO_AKSHARANTAR_MAPPING = {
    "en": "eng",
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "ml": "mal",
    "sn": "san",
    "pa": "pan",
    "as": "asm",
    "bn": "beng",
    "or": "ory",
    "ur": "urd",
    "ka": "kan",
    "gu": "guj",
}

AKSHARANTAR_TO_ULCA_LANGUAGE_CODE_MAPPING = {
    val: key for key, val in ULCA_LANGUAGE_CODE_TO_AKSHARANTAR_MAPPING.items()
}

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


DATASET_INPUT_COLUMN_MAPPING = {"facebook/flores": "sentence_"}

EnvSettings = _EnvSettings()
