import os
import sys
import copy
import time
import json
import logging
from glob import glob
from typing import Any, List, Union, Iterable, Dict, Tuple
from pathlib import Path
import base64
import wave
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm

from helpers import class_factory
from plugins import PluginBase
from config import BaseConfig
from plugins.scorers.config import (
    IndicTTSScorerConfig
)
from plugins.preprocessors.config import (
    IndicTTSConfig as IndicTTSPPConfig
)


class TTSScorer(PluginBase):
    """
    Plugin for TTS
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self._logger.debug(f"self.kwargs model: {self.kwargs}")
        self.model = self.kwargs["model"]()
        os.makedirs(self.tts_config.OUTPUT_DIR, exist_ok=True)

    def base64_to_wav(self, base64_string, filename):
        file_bytes = base64.b64decode(base64_string)
        with open(filename, "wb") as f:
            f.write(file_bytes)



        # # decode the base64 string
        # decoded_audio = base64.b64decode(base64_string)
        
        # # open a file in write mode
        # with wave.open(filename, 'wb') as audio_file:
        #     # set the parameters for the wave file
        #     audio_file.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
        #     # write the decoded audio data to the file
        #     audio_file.writeframes(decoded_audio)
        

    def read_inputs(self) -> Dict:
        with open(self.preprocessor_config.PREPROCESSED_FILE, "r") as ipf:
            for line in ipf:
                op = json.loads(line)
                yield op

    def populate_batch(self, batch_data, batch_data_lens):
        pass

    def get_inputs(self) -> Tuple[List, List, np.array, np.array, np.array]:
        batch_audio = []
        batch_filenames = []
        batch_text = []
        for data in self.read_inputs():
            batch_filenames.append(data["filename"])
            batch_text.append(data["transcript"])
            batch_audio.append(data["audio"])

            if len(batch_audio) == self.tts_config.BATCH_SIZE:
                yield batch_filenames, batch_text, batch_audio
                batch_audio = []
                batch_text = []
                batch_filenames = []
                # break

        if batch_audio:
            yield batch_filenames, batch_text, batch_audio

    def invoke(self, *args, **kwargs) -> Any:
        start_time = time.time()
        error_rows = []
        #with open(self.asr_config.OUTPUT_FILE, "w") as opf:
        for batch_filenames, batch_text, batch_correct_audio in tqdm(
            self.get_inputs(), total=self.preprocessor_config.NUM_INPUT_LINES
        ):
            try:
                results = [self.model.invoke(input=batch_text, model_type = "TTS")["audio"][0]["audioContent"]]
                #print(results)
            except Exception as e:
                print(e)
                error_rows.append([batch_text[0]])
                continue

            batch_results = []
            for filename, result in zip(batch_filenames, results):
                file_loc = os.path.join(self.tts_config.OUTPUT_DIR,filename.split("/")[-1])
                print(len(result))
                self.base64_to_wav(result, file_loc)

        pd.DataFrame(error_rows, columns=["Text"]).to_csv("error_files.csv")
        self._logger.info("Total Time Taken {}".format(time.time() - start_time))




IndicTTSScorer = class_factory(
    "IndicTTSScorer",
    (TTSScorer,),
    {"tts_config": IndicTTSScorerConfig(), "preprocessor_config": IndicTTSPPConfig()}
)