import os
import sys
import copy
import time
import json
import logging
from glob import glob
from typing import Any, List, Union, Iterable, Dict, Tuple
from pathlib import Path

import requests
import numpy as np
from tqdm import tqdm
import soundfile as sf
# import tritonclient.grpc as grpc_client
import tritonclient.http as http_client
from tritonclient.utils import InferenceServerException

from plugins import PluginBase
from config import BaseConfig
from plugins.scorers.config import ASRBatchInputValidation
from plugins.preprocessors.config import MUCSHindiConfig, CommonVoiceConfig, IndicSUPERBTestKnownConfig, IndicSUPERBTestUnknownConfig


class ASRBatchE2EScorer(PluginBase):
    """
    Plugin for Batch ASR
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        """
        # ToDo
        # Get from arg parse and pass in kwargs
        kwargs["scorer_config"] = {"MODE": "functional"}
        self.kwargs = kwargs
        self.model = self.kwargs["model"]()
        self.asr_config = ASRBatchInputValidation(**self.kwargs["scorer_config"])

        # This reference won't work. Change
        self.preprocessor_config = MUCSHindiConfig()

    def read_inputs(self) -> Dict:
        with open(self.preprocessor_config.PREPROCESSED_FILE, "r") as ipf:
            for line in ipf:
                # self._logger.debug(f"reading lines: {line}")
                yield json.loads(line)

    def populate_batch(self, batch_data, batch_data_lens):
        max_length = max(batch_data_lens)
        batch_size = len(batch_data)

        padded_zero_array = np.zeros((batch_size, max_length), dtype=np.uint8)
        for idx, data in enumerate(batch_data):
            padded_zero_array[idx, 0 : batch_data_lens[idx]] = data
        return padded_zero_array, np.reshape(batch_data_lens, [-1, 1]), np.array([0])  # batch dur for later

    def get_inputs(self) -> Tuple[List, np.array, np.array, np.array]:
        batch_data = []
        batch_data_lens = []
        batch_correct_text = []
        for data in self.read_inputs():
            batch_correct_text.append(data["transcript"])
            batch_data.append(data["audio"])
            batch_data_lens.append(int(len(data["audio"])))

            if len(batch_data) == self.asr_config.BATCH_SIZE:
                padded_zero_array, batch_audio_len, batch_data_dur = self.populate_batch(batch_data, batch_data_lens)
                batch_data = []
                batch_data_lens = []
                bct = copy.deepcopy(batch_correct_text)
                batch_correct_text = []
                yield bct, padded_zero_array, batch_audio_len, batch_data_dur

        if batch_data:
            padded_zero_array, batch_audio_len, batch_data_dur = self.populate_batch(batch_data, batch_data_lens)
            yield batch_correct_text, padded_zero_array, batch_audio_len, batch_data_dur

    def invoke(self, *args, **kwargs) -> Any:
        """
        """
        start_time = time.time()
        with open(self.asr_config.OUTPUT_FILE, "w") as opf:
            for batch_correct_text, batch_audio_raw, batch_audio_len, batch_data_dur in tqdm(
                self.get_inputs(), total=self.preprocessor_config.NUM_INPUT_LINES
            ):
                results = self.model.invoke(input=batch_audio_raw)

                batch_results = []
                for result, correct_text in zip(results, batch_correct_text):
                    self._logger.info(f"{result}, {correct_text}")
                    batch_results.append(json.dumps({"output": result, "transcript": correct_text}) + "\n")
                opf.writelines(batch_results)

        total_time = time.time() - start_time
        self._logger.info("Total Time Taken {}".format(total_time))
        # self._logger.info("RTFX is: {}".format(self.asr_config.ITERATIONS * np.sum(batch_data_dur.ravel()) / total_time))

        # statistics = self.model.triton_client.get_inference_statistics(
        #     model_name=self.model.asr_config.MODEL_NAME,
        #     headers=self.model.headers
        # )
        # self._logger.info(json.dumps(statistics, indent=4))


