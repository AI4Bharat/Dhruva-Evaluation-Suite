import os
from typing import List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseDatasetConfig():
    DATASET_URL: str
    DATASET_VERSION: int
    DATASET_NAME: str
    LOCAL_PATH: Path


@dataclass
class IndicSUPERBTestKnownConfig(BaseDatasetConfig):
    DATASET_URL: str = "https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testkn_audio.tar"
    DATASET_VERSION: int = 1
    DATASET_NAME: str = "IndicSUPERB - Test Known"
    LOCAL_PATH: Path = Path("../datasets/raw/IndicSUPERB/test-known.tar")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)


@dataclass
class IndicSUPERBTestUnknownConfig(BaseDatasetConfig):
    DATASET_URL: str = "https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testunk_audio.tar"
    DATASET_VERSION: int = 1
    DATASET_NAME: str = "IndicSUPERB - Test Unknown"
    LOCAL_PATH: Path = Path("../datasets/raw/IndicSUPERB/test-unknown.tar")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)

@dataclass
class IndicSUPERBTestTranscriptsConfig(BaseDatasetConfig):
    DATASET_URL: str = "https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/transcripts_n2w.tar"
    DATASET_VERSION: int = 1
    DATASET_NAME: str = "IndicSUPERB - Test Transcripts"
    LOCAL_PATH: Path = Path("../datasets/raw/IndicSUPERB/transcripts.tar")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)

@dataclass
class MUCSHindiConfig(BaseDatasetConfig):
    DATASET_URL: str = "http://openslr.elda.org/resources/103/Hindi_test.tar.gz"
    DATASET_VERSION: int = 1
    DATASET_NAME: str = "MUCS - Test"
    LOCAL_PATH: Path = Path("../datasets/raw/MUCS/test.tar.gz")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)
