import os
import sys
import copy
import time
import json
import logging
from glob import glob
from typing import Any, List, Union, Iterable, Dict, Tuple
from pathlib import Path

import pandas as pd
import requests
import numpy as np
from tqdm import tqdm
import soundfile as sf
import tritonclient.http as http_client
from tritonclient.utils import InferenceServerException

from helpers import class_factory
from plugins import PluginBase
from config import BaseConfig
from plugins.scorers.config import (
    MUCSScorerConfig,
    CommonVoiceScorerConfig,
    IndicSUPERBTestKnownScorerConfig,
    IndicSUPERBTestUnKnownScorerConfig
)
from plugins.preprocessors.config import (
    MUCSHindiConfig as MUCSHindiPPConfig,
    CommonVoiceConfig as CommonVoicePPConfig,
    IndicSUPERBTestKnownConfig as IndicSUPERBTestKnownPPConfig,
    IndicSUPERBTestUnknownConfig as IndicSUPERBTestUnknownPPConfig
)


class ASRBatchOffConfScorer(PluginBase):
    """
    Plugin for Batch ASR
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self._logger.debug(f"self.kwargs model: {self.kwargs}")
        self.model = self.kwargs["model"]()

    def read_inputs(self) -> Dict:
        with open(self.preprocessor_config.PREPROCESSED_FILE, "r") as ipf:
            for line in ipf:
                op = json.loads(line)
                yield op

    def populate_batch(self, batch_data, batch_data_lens):
        pass

    def get_inputs(self) -> Tuple[List, List, np.array, np.array, np.array]:
        batch_data = []
        batch_filenames = []
        batch_correct_text = []
        for data in self.read_inputs():
            batch_filenames.append(data["filename"])
            batch_correct_text.append(data["transcript"])
            batch_data.append(data["audio"])

            if len(batch_data) == self.asr_config.BATCH_SIZE:
                yield batch_filenames, batch_correct_text, batch_data
                batch_data = []
                batch_correct_text = []
                batch_filenames = []
                # break

        if batch_data:
            yield batch_filenames, batch_correct_text, batch_data

    def invoke(self, *args, **kwargs) -> Any:
        start_time = time.time()
        error_rows = []
        with open(self.asr_config.OUTPUT_FILE, "w") as opf:
            for batch_filenames, batch_correct_text, batch_audio_raw in tqdm(
                self.get_inputs(), total=self.preprocessor_config.NUM_INPUT_LINES
            ):
                try:
                    results = self.model.invoke(input=batch_audio_raw, model_type = "ASR")["data"]
                except Exception as e:
                    # print(e)
                    error_rows.append([os.path.basename(batch_filenames[0]), batch_audio_raw[0]])
                    continue

                batch_results = []
                for filename, result, correct_text in zip(batch_filenames, results, batch_correct_text):
                    self._logger.info(f"{os.path.basename(filename)} {result} {type(result)}")
                    batch_results.append(json.dumps({"output": result, "transcript": correct_text}) + "\n")
                opf.writelines(batch_results)
        pd.DataFrame(error_rows, columns=["filename", "audio"]).to_csv("error_files.csv")
        self._logger.info("Total Time Taken {}".format(time.time() - start_time))


# Dataset specific scorers
MUCSBatchOffConfScorer = class_factory(
    "MUCSBatchOffConfScorer",
    (ASRBatchOffConfScorer,),
    {"asr_config": MUCSScorerConfig(), "preprocessor_config": MUCSHindiPPConfig()}
)


CommonVoiceBatchOffConfScorer = class_factory(
    "CommonVoiceBatchOffConfScorer",
    (ASRBatchOffConfScorer,),
    {"asr_config": CommonVoiceScorerConfig(), "preprocessor_config": CommonVoicePPConfig()}
)


IndicSUPERBKnownBatchOffConfScorer = class_factory(
    "IndicSUPERBKnownBatchOffConfScorer",
    (ASRBatchOffConfScorer,),
    {"asr_config": IndicSUPERBTestKnownScorerConfig(), "preprocessor_config": IndicSUPERBTestKnownPPConfig()}
)

IndicSUPERBUnknownBatchOffConfScorer = class_factory(
    "IndicSUPERBUnknownBatchOffConfScorer",
    (ASRBatchOffConfScorer,),
    {"asr_config": IndicSUPERBTestUnKnownScorerConfig(), "preprocessor_config": IndicSUPERBTestUnknownPPConfig()}
)