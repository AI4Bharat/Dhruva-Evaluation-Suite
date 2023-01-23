import os
from pathlib import Path
from typing import List, Literal
from dataclasses import dataclass, field, make_dataclass


@dataclass
class BaseScorerConfig():
    BATCH_SIZE: int = 1
    ITERATIONS: int = 10
    MODE: Literal["performance", "functional"]= "functional"
    OUTPUT_FILE: str = "../datasets/outputs/{}/Hindi/test/output.jsonl"
    

    def __post_init__(self):
        Path(os.path.dirname(self.OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)


# @dataclass
# class ASRBatchInputValidation(BaseScorerConfig):
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
    bases=(BaseScorerConfig,),
    # namespace={'add_one': lambda self: self.x + 1}
)

CommonVoiceScorerConfig = make_dataclass(
    'CommonVoiceScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("CommonVoice"))),
    ],
    bases=(BaseScorerConfig,)
)


IndicSUPERBTestKnownScorerConfig = make_dataclass(
    'IndicSUPERBTestKnownScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("IndicSUPERB_test_known"))),
    ],
    bases=(BaseScorerConfig,)
)


IndicSUPERBTestUnKnownScorerConfig = make_dataclass(
    'IndicSUPERBTestUnKnownScorer',
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/Hindi/test/output.jsonl".format("IndicSUPERB_test_unknown"))),
    ],
    bases=(BaseScorerConfig,)
)


FLORES200ScorerConfig = make_dataclass(
    "FLORES200NMTScorerConfig",
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/test/output.jsonl".format("flores200"))),
    ],
    bases=(BaseScorerConfig,)
)


WAT20ScorerConfig = make_dataclass(
    "WAT20ScorerConfig",
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/test/output.jsonl".format("wat20"))),
    ],
    bases=(BaseScorerConfig,)
)


WAT21ScorerConfig = make_dataclass(
    "WAT21ScorerConfig",
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/test/output.jsonl".format("wat21"))),
    ],
    bases=(BaseScorerConfig,)
)


WMTScorerConfig = make_dataclass(
    "WMTScorerConfig",
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/test/output.jsonl".format("wmt"))),
    ],
    bases=(BaseScorerConfig,)
)


UFALSScorerConfig = make_dataclass(
    "UFALSScorerConfig",
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/test/output.jsonl".format("ufals"))),
    ],
    bases=(BaseScorerConfig,)
)


PMIScorerConfig = make_dataclass(
    "PMIScorerConfig",
    [
        ('OUTPUT_FILE', str, field(default="../datasets/outputs/{}/test/output.jsonl".format("pmi"))),
    ],
    bases=(BaseScorerConfig,)
)
