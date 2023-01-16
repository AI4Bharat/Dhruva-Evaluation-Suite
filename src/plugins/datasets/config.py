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
class MUCSHindiConfig(BaseDatasetConfig):
    DATASET_URL: str = "http://openslr.elda.org/resources/103/Hindi_test.tar.gz"
    DATASET_VERSION: int = 1
    DATASET_NAME: str = "MUCS - Test"
    LOCAL_PATH: Path = Path("../datasets/raw/MUCS/test.tar.gz")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)

@dataclass
class CommonVoiceConfig(BaseDatasetConfig):
    DATASET_URL: str = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-12.0-2022-12-07/cv-corpus-12.0-2022-12-07-en.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3PT7OZ2RB%2F20230107%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230107T212122Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEAcaDKfrOZQn91YKQB66FyKSBJFddH0aOA8lwaS0eDgkjwyjeAlXgW9XBkk5OWFTLyGH%2Bhuf%2BLBQBtxvjzPV6gXj50kKSn9HDAvJWwrBpnPDUuJo4f3rstohO9%2BhfvDKRx%2Bzhes6CE3mNsX7wlE0p%2BQrxWRo%2FnqQUukF6wXnivSoKBre7eBTe3CsgN7AiQX%2BhuTuuqCoMmEGmaujStO0kJ3%2FEwPtooQGogAdvRvuzFcQ1ecRgNLn84keH37xiF%2FlQhXpgRaiVO6gSzWdyqhyBOpNx0FDIvJqSTLcd6pqFhdK2xGJppp72yNHjCy94DIHMF1GpVWSSJvoQgCbpYuP47vUowfK9jTO2Ci5qrQxWzdZuawEFdDlmvVkJEaBLL%2FGMKfvSjTEEW0GdWermi2TJBPBvBQGPY1cxrmkftlGJzBvKso0fsSRWEkF5eT4rgNJVLcSuX0qxZhU7XXqy9g4t%2FEmBei7uGgT0FVfMXLpx4Xz9mf6942wmPpAAanMQpSCaXz%2B7at5KcLh1g9I78l9Xpdgw1AQfLmb8pqRfXdYuAC5ZAvLvj0argK%2FYmeNO6Tjwxc7PoDJDoO7eNVxgfviODzr2ZMGrxkz3O1ZSYcTIt%2FWqb5n%2B1JJ1t%2B91Bm%2BaMhRmag9lbvu7vHo2vzB%2FCIb2Xr5l8oEWeOBnJmXn6xyR7%2Bgp8I7AdU48DDGbAgWG6Tamda3nI8XimaacsA%2BeTNwpe0UWLjFKMTA550GMipt2y0L6k8zBJEJToljaSm0WWViyK0ChgOWB5NYGACjIZAm3y6uc4TaOEQ%3D&X-Amz-Signature=aec726f945ff9b71a7e1de158a0f655280dfb1e5298bce697231ba6ba5cb5d01&X-Amz-SignedHeaders=host"
    DATASET_VERSION: int = 12
    DATASET_NAME: str = "CommonVoice Corpus 12"
    LOCAL_PATH: Path = Path("../datasets/raw/CommonVoice/test.tar.gz")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)
