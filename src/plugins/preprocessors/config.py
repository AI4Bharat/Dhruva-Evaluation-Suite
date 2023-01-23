import os
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass, field


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


@dataclass
class FLORES200PreProcessorConfig(BaseDatasetConfig):
    DATASET_TYPE: str = "multilingual"
    EXTRACT_PATH: Path = Path("../datasets/raw/flores200_dataset")
    INPUT_TEST_PATH: Path = Path("../datasets/raw/flores200_dataset/")
    INPUT_TEST_FILE: str = "test"
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/flores200_dataset/preprocessed.jsonl")
    LANG_CODE_LOOKUP: Dict[str, str] = field(default_factory=lambda: {
        "as": "asm_Beng",
        "bn": "ben_Beng",
        "gu": "guj_Gujr",
        "hi": "hin_Deva",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "mr": "mar_Deva",
        "or": "ory_Orya",
        "pa": "pan_Guru",
        "ta": "tam_Taml",
        "te": "tel_Telu",
    })
    LANGS: List = field(default_factory=lambda: ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"])
    BATCH_SIZE: int = 100
    MODE: str = "en-indic"

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)


@dataclass
class WAT20PreProcessorConfig(BaseDatasetConfig):
    DATASET_TYPE: str = "bilingual"
    EXTRACT_PATH: Path = Path("../datasets/raw/wat20/")
    INPUT_TEST_FILE: str = "test"
    INPUT_TEST_PATH: Path = Path("../datasets/raw/wat20/benchmarks/wat2020-devtest/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/wat20/benchmarks/wat20-devtest/preprocessed.jsonl")
    LANGS: List = field(default_factory=lambda: ["bn", "gu", "hi", "ml", "mr", "ta", "te"])
    BATCH_SIZE: int = 100
    MODE: str = "en-indic"

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)


@dataclass
class WAT21PreProcessorConfig(BaseDatasetConfig):
    DATASET_TYPE: str = "multilingual"
    EXTRACT_PATH: Path = Path("../datasets/raw/wat21/")
    INPUT_TEST_FILE: str = "test"
    INPUT_TEST_PATH: Path = Path("../datasets/raw/wat21/benchmarks/wat2021-devtest/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/wat21/benchmarks/wat2021-devtest/preprocessed.jsonl")
    LANGS: List = field(default_factory=lambda: ["bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"])
    BATCH_SIZE: int = 100
    MODE: str = "en-indic"

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)


@dataclass
class WMTPreProcessorConfig(BaseDatasetConfig):
    DATASET_TYPE: str = "bilingual"
    EXTRACT_PATH: Path = Path("../datasets/raw/wmt/")
    INPUT_TEST_FILE: str = "test"
    INPUT_TEST_PATH: Path = Path("../datasets/raw/wmt/benchmarks/wmt-news/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/wmt/benchmarks/wmt-news/preprocessed.jsonl")
    LANGS: List = field(default_factory=lambda: ["gu", "hi", "ta"])
    # LANGS: List = field(default_factory=lambda: ["hi", "ta"])
    BATCH_SIZE: int = 100
    MODE: str = "en-indic"

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)


@dataclass
class UFALSPreProcessorConfig(BaseDatasetConfig):
    DATASET_TYPE: str = "bilingual"
    EXTRACT_PATH: Path = Path("../datasets/raw/ufals/")
    INPUT_TEST_FILE: str = "test"
    INPUT_TEST_PATH: Path = Path("../datasets/raw/ufals/benchmarks/ufals-ta/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/ufals/benchmarks/ufals-ta/preprocessed.jsonl")
    LANGS: List = field(default_factory=lambda: ["ta"])
    BATCH_SIZE: int = 100
    MODE: str = "en-indic"

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)


@dataclass
class PMIPreProcessorConfig(BaseDatasetConfig):
    DATASET_TYPE: str = "bilingual"
    EXTRACT_PATH: Path = Path("../datasets/raw/pmi/")
    INPUT_TEST_FILE: str = "test"
    INPUT_TEST_PATH: Path = Path("../datasets/raw/pmi/benchmarks/pmi/")
    PREPROCESSED_FILE: Path = Path("../datasets/preprocessed/pmi/benchmarks/pmi/preprocessed.jsonl")
    LANGS: List = field(default_factory=lambda: ["as"])
    BATCH_SIZE: int = 100
    MODE: str = "en-indic"

    def __post_init__(self):
        Path(os.path.dirname(self.EXTRACT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.PREPROCESSED_FILE)).mkdir(parents=True, exist_ok=True)
