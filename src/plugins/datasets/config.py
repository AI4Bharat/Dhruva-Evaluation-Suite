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
    DATASET_URL: str = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-12.0-2022-12-07/cv-corpus-12.0-2022-12-07-hi.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3FK7VF5KR%2F20230117%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230117T191146Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEPT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDNQlFRF2bSCIewoyqiKSBBCpVSvVcBaXWnpUXigsShfG3y62MYxIJ9hdFHp0ozvkOUIpky%2BHDzaK0niS5LpL62vMgkxL12gjgMk7Sba9lsv%2B%2Bnfwly8hWJZB5o7RHGqKwvtcitIjPbHxewUp%2B9Wyftt8A430CuvZP1b%2BmK7JLTROt0ouEu1%2B84lRP9kV%2FAUyeGR4dvRXn0EgNv%2F6OrWgyfV%2FSCqAHoeoCD1j6%2B0sM6SZU3S2YrUIm8OpNRGSqblYqgitAfOKi%2B6LW48kQQqF0e%2BwgZtL%2BMDnZSq%2Fx8SZcLU6ISeAXNAjCTFfAx7QVcZmmhdVdDYZF5uvfXNw8EP92aqK67Y7%2F9wGdiRn2eqXD7lRRYV%2BBT1lPFRA%2B606JnSHrqALL%2F2%2FaiV%2BjWKy8TeMUIsavXRJVa1yIgOKN%2FAM%2BO0jyk3FbWikBQnuA46sePo995w6p9vynj3z0xukvG2HegSr1zfRIvGwkcO6nZEhLLQmVSbBPH9Hvl2FuZaUNoMCGBof14wTTWur%2F6sKUUXziSGyBe7ee59WZJeEa6CrRxUDZMT5CvMcsJjIbrZGTbX4MgC0gOK7QQrSTzBkifVoDwRgpqsnYSZUMqhCH2R5C8Ln0UkiEiYiXgP9W64L22KZatO4iTo1z77ENaYwx4gu%2BqoKgaRgGzvBsa5BtHE2g2rMna6RofHVmjLdAIG8cYXuGsAGFk6JJDNYHVyyGLwvCu%2F9KNfUm54GMippzd%2BgoasKFbZOS7q2TLcee8TM5BgFj7u9xG8TfgD7uqH%2BaYIjDTQxVmU%3D&X-Amz-Signature=d8d3323388a6995aa718b1f21ba3f326e7d504924eff74bc7692e732eeaa6e17&X-Amz-SignedHeaders=host"
    DATASET_VERSION: int = 12
    DATASET_NAME: str = "CommonVoice Corpus 12"
    LOCAL_PATH: Path = Path("../datasets/raw/CommonVoice/test.tar.gz")

    def __post_init__(self):
        Path(os.path.dirname(self.LOCAL_PATH)).mkdir(parents=True, exist_ok=True)
