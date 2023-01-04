import os
from typing import List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseDatasetConfig():
    EXTRACT_PATH: Path


@dataclass
class IndicSUPERBTestKnownConfig(BaseDatasetConfig):
    EXTRACT_PATH: Path = Path("../datasets/raw/IndicSUPERB/test_known/")

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)


@dataclass
class MUCSHindiConfig(BaseDatasetConfig):
    EXTRACT_PATH: Path = Path("../datasets/raw/MUCS/Hindi/")
    INPUT_TRANSCRIPT_FILE: Path = Path("../datasets/raw/MUCS/Hindi/test/transcription.txt")
    INPUT_AUDIO_FILES: Path = Path("../datasets/raw/MUCS/Hindi/test/audio/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/MUCS/Hindi/test/preprocessed.jsonl")
    LANG_CODE: str = "hi"
    BATCH_SIZE: int = 100

    NUM_INPUT_LINES: int = 0

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)
