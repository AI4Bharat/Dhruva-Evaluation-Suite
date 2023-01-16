import os
from pathlib import Path
from typing import List, Literal
from dataclasses import dataclass, field, make_dataclass


@dataclass
class BaseConfig():
    BATCH_SIZE: int = 1
    ITERATIONS: int = 10
    MODE: Literal["performance", "functional"]= "functional"
    OUTPUT_FILE: str = "../datasets/outputs/{}/Hindi/test/output.jsonl"
    

    def __post_init__(self):
        Path(os.path.dirname(self.OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)

        if self.MODE == "functional":
            self.ITERATIONS = 1


# @dataclass
# class ASRBatchInputValidation(BaseConfig):
#     BATCH_SIZE: int = 1
#     ITERATIONS: int = 10
#     # OUTPUT_FILE: Path = Path("")


# All dataclasses are present to give useful defaults for specific dataset and model combinations
# Override via CLI / YAML if needed

MUCSScorerConfig = make_dataclass(
    'MUCSScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("MUCS"))),
    ],
    bases=(BaseConfig,),
    # namespace={'add_one': lambda self: self.x + 1}
)

CommonVoiceScorerConfig = make_dataclass(
    'CommonVoiceScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("CommonVoice"))),
    ],
    bases=(BaseConfig,)
)


IndicSUPERBTestKnownScorerConfig = make_dataclass(
    'IndicSUPERBTestKnownScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("IndicSUPERB_test_known"))),
    ],
    bases=(BaseConfig,)
)


IndicSUPERBTestUnKnownScorerConfig = make_dataclass(
    'IndicSUPERBTestUnKnownScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("IndicSUPERB_test_unknown"))),
    ],
    bases=(BaseConfig,)
)
