import os
from typing import List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseDatasetConfig():
    EXTRACT_PATH: Path
    NUM_INPUT_LINES: int = 0


@dataclass
class IndicSUPERBTestKnownConfig(BaseDatasetConfig):
    EXTRACT_PATH: Path = Path("../datasets/raw/IndicSUPERB/test_known/")

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)


@dataclass
class MUCSHindiConfig(BaseDatasetConfig):
    EXTRACT_PATH: Path = Path("../datasets/raw/MUCS/Hindi/")
    INPUT_TRANSCRIPT_FILE: Path = Path("../datasets/raw/MUCS/Hindi/test/transcription.txt")
    INPUT_AUDIO_FILES: Path = Path("../datasets/raw/MUCS/Hindi/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/MUCS/Hindi/test/preprocessed.jsonl")
    LANG_CODE: str = "hi"
    BATCH_SIZE: int = 100

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)

@dataclass
class CommonVoiceConfig(BaseDatasetConfig):
    EXTRACT_PATH: Path = Path("../datasets/raw/CommonVoice/Hindi/")
    INPUT_TRANSCRIPT_FILE: Path = Path("../datasets/raw/CommonVoice/hi/test.tsv")
    INPUT_AUDIO_FILES: Path = Path("../datasets/raw/CommonVoice/hi/clips/")
    INPUT_WAVAUDIO_FILES: Path = Path("../datasets/raw/CommonVoice/hi/audio_wav/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/CommonVoice/Hindi/preprocessed.jsonl")
    LANG_CODE: str = "hi"
    BATCH_SIZE: int = 100

    NUM_INPUT_LINES: int = 0

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)
