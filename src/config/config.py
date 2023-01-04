import logging
from typing import List
from dataclasses import dataclass


@dataclass
class BaseConfig():
    LOGGER: str = __name__
    LOGGING_LEVEL: str = logging.DEBUG
