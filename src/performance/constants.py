from pydantic import BaseSettings as PydanticBaseSettings, BaseModel
from typing import Union


# User config
class _Task(BaseModel):
    name: str
    type: str


class _Model(BaseModel):
    url: str
    type: str


class _Params(BaseModel):
    payload_path: str
    test_params: dict[int, dict[str, Union[int, str]]]


class UserConfiguration(PydanticBaseSettings):
    """Initialises all user configs from a yaml file"""

    model: _Model
    task: _Task
    params: _Params
