import os
import json
import tarfile
import logging
from pathlib import Path
from typing import List, Any, Dict
from tqdm import tqdm
import numpy as np
import soundfile as sf

from plugins import PluginBase
from config import BaseConfig
from plugins.preprocessors.speech_transcript_cleaning import cleaning_pipeline, get_dict_chars
from plugins.preprocessors.config import IndicTTSConfig

class TTSPreProcessor(PluginBase):
    """
    - Remove puctuations
    - Normalize the characters
    - Convert num to word
    """

    def __init__(self, **kwargs) -> None:
        """
        :param kwargs:
        """
        self.kwargs = kwargs
        self.config = None

    # def load_wav(self, path: Path) -> np.array:
    #     return np.fromfile(path, dtype="uint8").tolist()

    def load_wav(self, path: Path):
        audio, _ = sf.read(path)
        return audio.tolist()

    def preprocess(self):
        """
        param: sentences - List of sentences to preprocess
        param: audios - List of file paths for audios
        """
        dict_characters = get_dict_chars(self.config.LANG_CODE)
        # Read input and preprocess it line by line. Batch it while writing
        for raw_audio, sentence in self.get_inputs():
            no_ood, preprocessed_sentence = cleaning_pipeline(dict_characters, sentence, self.config.LANG_CODE)
            audio = self.load_wav(raw_audio)
            yield no_ood, {"filename": raw_audio, "audio": audio, "transcript": preprocessed_sentence}

    def write_preprocessed_output(self):
        total_sents = 0
        count_no_ood = 0
        batch = []
        with open(self.config.PREPROCESSED_FILE, "w") as ppf:
            # for no_ood, intermediate_output in self.preprocess():
            # with tqdm(total=100) as pbar:
            for no_ood, intermediate_output in tqdm(self.preprocess(), total=self.config.NUM_INPUT_LINES):
                batch.append(json.dumps(intermediate_output) + "\n")
                total_sents += 1
                count_no_ood += no_ood

                # Batching while writing to improve throughput
                if len(batch) == self.config.BATCH_SIZE:
                    ppf.writelines(batch)
                    batch = []

                    # pbar.update(total_sents / self.num_input_lines * 100)
            ppf.writelines(batch)

        self._logger.info(f"---------- Preprocessing completed ----------")
        self._logger.info(f"Total sentences = {total_sents} and total OOD sentences = {count_no_ood}")

    def invoke(self, *args, **kwargs) -> Any:
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :param kwargs: possible keyword arguments for the plugin
        :return: None
        """

        if os.path.exists(self.config.PREPROCESSED_FILE):
            return self.config.PREPROCESSED_FILE

        self.write_preprocessed_output()
        return self.config.PREPROCESSED_FILE

    def get_inputs(self, *args, **kwargs):
        raise NotImplementedError


class IndicTTSPreProcessor(TTSPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # intialising configs here to be able to override via kwargs
        self.config = IndicTTSConfig()
        self._logger.info("Calculating lines in file ...")
        self.config.NUM_INPUT_LINES = int(sum(1 for line in open(self.config.INPUT_TRANSCRIPT_FILE)))
        self._logger.info(f"Number of lines: {self.config.NUM_INPUT_LINES}")

    # def invoke(self, *args, **kwargs):
    #     pass

    def get_inputs(self, *args, **kwargs):
        with open(self.config.INPUT_TRANSCRIPT_FILE, "r") as read_fp:
            for line in read_fp:
                elements = line.split("|")
                # print(" ".join(elements[1:]))
                yield os.path.join(self.config.INPUT_AUDIO_FILES, elements[0] + ".wav"), elements[1]
                # break