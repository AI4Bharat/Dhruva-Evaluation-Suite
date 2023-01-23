import os
import json
import tarfile
import logging
from pathlib import Path
from typing import List, Any, Dict, Tuple

from tqdm import tqdm
import numpy as np
import soundfile as sf

import helpers
from plugins import PluginBase
from config import BaseConfig
from plugins.preprocessors.speech_transcript_cleaning import cleaning_pipeline, get_dict_chars
import plugins.preprocessors.config as preprocessor_config


class NMTPreProcessorBase(PluginBase):
    """
    - Read from source and Target Languages
    - Iterate through all languages
    - Populate a JSONL file
    """

    def __init__(self, **kwargs) -> None:
        """
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.config = self.config(**kwargs.get("user_config").get("preprocessor_config"))

    # def __init_subclass__(cls, **kwargs):
    #     # Init config
    #     # Enable user override using kwargs

    #     super().__init_subclass__(**kwargs)
    #     print("--> ", cls.config)
    #     if cls.config and "preprocessor_config" in kwargs and kwargs["preprocessor_config"]:
    #         cls.config(**kwargs["preprocessor_config"])

    @staticmethod
    def get_src_tgt_lang_codes(lang_code, mode):
        if mode == "en-indic":
            return "en", lang_code
        elif mode == "indic-en":
            return lang_code, "en"
        else:
            raise ValueError(f"Wrong mode {mode}")

    def preprocess(self):
        """
        param: sentences - List of sentences to preprocess
        param: audios - List of file paths for audios
        """
        if self.config.MODE in ("en-indic", "indic-en"):
            for lang_code in tqdm(self.config.LANGS):
                src_lang_code, tgt_lang_code = self.get_src_tgt_lang_codes(lang_code, self.config.MODE)

                for source_sentence, target_sentence in self.get_inputs(
                    src_lang_code=src_lang_code, tgt_lang_code=tgt_lang_code
                ):
                    # clean()
                    yield {
                        "source_sentence": source_sentence,
                        "target_sentence": target_sentence,
                        "source_lang": src_lang_code,
                        "target_lang": tgt_lang_code
                    }

    def write_preprocessed_output(self):
        total_sents = 0
        batch = []
        with open(self.config.PREPROCESSED_FILE, "w") as ppf:
            for intermediate_output in self.preprocess():
                batch.append(json.dumps(intermediate_output) + "\n")
                total_sents += 1

                # Batching while writing to improve throughput
                if len(batch) == self.config.BATCH_SIZE:
                    ppf.writelines(batch)
                    batch = []
            ppf.writelines(batch)

        self._logger.info(f"---------- Preprocessing completed ----------")
        self._logger.info(f"Total sentences = {total_sents}")

    def invoke(self, *args, **kwargs) -> str:
        # if os.path.exists(self.config.PREPROCESSED_FILE):
        #     self._logger.info("Preprocessed file exists! Skipping preprocessing")
        #     return self.config.PREPROCESSED_FILE

        helpers.extract_files(self.kwargs["dataset_output"], self.config.EXTRACT_PATH)
        self.write_preprocessed_output()
        return self.config.PREPROCESSED_FILE

    def get_inputs(self, *args, **kwargs):
        input_file = ""
        input_ref_file = ""

        if self.config.DATASET_TYPE == "bilingual":
            folder_name = ""
            if kwargs["src_lang_code"] == "en":
                folder_name = kwargs["src_lang_code"] + "-" + kwargs["tgt_lang_code"]
            elif kwargs["tgt_lang_code"] == "en":
                folder_name = kwargs["tgt_lang_code"] + "-" + kwargs["src_lang_code"]
            else:
                raise ValueError(f'Wrong src_lang_code: {kwargs["src_lang_code"]} or tgt_lang_code: {kwargs["tgt_lang_code"]}')

            input_file = os.path.join(
                self.config.INPUT_TEST_PATH, folder_name, self.config.INPUT_TEST_FILE + "." + kwargs["src_lang_code"]
            )
            input_ref_file = os.path.join(
                self.config.INPUT_TEST_PATH, folder_name, self.config.INPUT_TEST_FILE + "." + kwargs["tgt_lang_code"]
            )

        elif self.config.DATASET_TYPE == "multilingual":
            if self.config.NAME == "FLORES200":
                input_file = os.path.join(
                    self.config.INPUT_TEST_PATH, self.config.LANG_CODE_LOOKUP.get(kwargs["src_lang_code"]) + ".devtest"
                )
                input_ref_file = os.path.join(
                    self.config.INPUT_TEST_PATH, self.config.LANG_CODE_LOOKUP.get(kwargs["tgt_lang_code"]) + ".devtest"
                )
            else:
                input_file = os.path.join(
                    self.config.INPUT_TEST_PATH, self.config.INPUT_TEST_FILE + "." + kwargs["src_lang_code"]
                )
                input_ref_file = os.path.join(
                    self.config.INPUT_TEST_FILE + "." + kwargs["tgt_lang_code"]
                )

        else:
            raise ValueError(f"Wrong dataset_type {self.config.DATASET_TYPE}")

        with open(input_file, "r") as read_t, open(input_ref_file, "r") as read_r:
            for test, reference in tqdm(zip(read_t, read_r)):
                yield test, reference


FLORES200PreProcessor = helpers.class_factory(
    "FLORES200PreProcessor", (NMTPreProcessorBase,), {"config": preprocessor_config.FLORES200PreProcessorConfig}
)

WMTPreProcessor = helpers.class_factory(
    "WMTPreProcessor", (NMTPreProcessorBase,), {"config": preprocessor_config.WMTPreProcessorConfig}
)
WAT20PreProcessor = helpers.class_factory(
    "WAT20PreProcessor", (NMTPreProcessorBase,), {"config": preprocessor_config.WAT20PreProcessorConfig}
)

WAT21PreProcessor = helpers.class_factory(
    "WAT21PreProcessor", (NMTPreProcessorBase,), {"config": preprocessor_config.WAT21PreProcessorConfig}
)

UFALSPreProcessor = helpers.class_factory(
    "UFALSPreProcess or", (NMTPreProcessorBase,), {"config": preprocessor_config.UFALSPreProcessorConfig}
)

PMIPreProcessor = helpers.class_factory(
    "PMIPreProcessor", (NMTPreProcessorBase,), {"config": preprocessor_config.PMIPreProcessorConfig}
)
