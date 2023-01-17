import os
from pathlib import Path
from typing import List, Literal
from dataclasses import dataclass, field


@dataclass
class BaseConfig():
    BATCH_SIZE: int
    ITERATIONS: int
    MODE: Literal["performance", "functional"]= "functional"
    OUTPUT_FILE: Path = Path("")

    def __post_init__(self):
        Path(os.path.dirname(self.OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)

        if self.MODE == "functional":
            self.ITERATIONS = 1


@dataclass
class ASRBatchInputValidation(BaseConfig):
    BATCH_SIZE: int = 1
    ITERATIONS: int = 10
    OUTPUT_FILE: Path = Path("../datasets/outputs/MUCS/test/Hindi/output.jsonl")
