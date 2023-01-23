import os
import time
import json
import logging
from time import sleep
from pathlib import Path
import multiprocessing as mp
from typing import Any, List, Union, Iterable, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from helpers import class_factory
from plugins import PluginBase
from config import BaseConfig
from plugins.scorers.config import (
    FLORES200ScorerConfig,
    WAT20ScorerConfig,
    WAT21ScorerConfig,
    WMTScorerConfig,
    UFALSScorerConfig,
    PMIScorerConfig
)
from plugins.preprocessors.config import (
    FLORES200PreProcessorConfig,
    WAT20PreProcessorConfig,
    WAT21PreProcessorConfig,
    WMTPreProcessorConfig,
    UFALSPreProcessorConfig,
    PMIPreProcessorConfig
)

num_cores = 1  # mp.cpu_count()


class NMTScorer(PluginBase):
    """
    Plugin for Batch ASR
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.model = self.kwargs["model"]()
        self._logger.debug(f"model: {self.kwargs['model']}")

    def read_inputs(self) -> Dict:
        with open(self.preprocessor_config.PREPROCESSED_FILE, "r") as ipf:
            for line in ipf:
                op = json.loads(line)
                yield op

    def get_inputs(self) -> Tuple[List, List, List, List]:
        batch_data = []
        batch_correct_text = []
        batch_source_langs = []
        batch_target_langs = []
        for data in self.read_inputs():
            batch_data.append(data["source_sentence"])
            batch_correct_text.append(data["target_sentence"])
            batch_source_langs.append(data["source_lang"])
            batch_target_langs.append(data["target_lang"])

            if len(batch_data) == self.config.BATCH_SIZE:
                yield batch_correct_text, batch_data, batch_source_langs, batch_target_langs
                batch_data = []
                batch_correct_text = []
                batch_source_langs = []
                batch_target_langs = []
                # break

        if batch_data:
            yield batch_correct_text, batch_data, batch_source_langs, batch_target_langs

    def invoke(self, *args, **kwargs) -> Any:
        start_time = time.time()
        error_rows = []
        self._logger.info("Scoring...")

        # manager = mp.Manager()
        # read_q = mp.Queue()  # Should it be bounded?  maxsize=num_cores
        # write_q = mp.Queue()  # Should it be bounded?
        # iolock = mp.Lock()

        # self._logger.debug("queueing started")
        # for batch_correct_text, batch_data, batch_source_langs, batch_target_langs in tqdm(self.get_inputs()):
        #     # self._logger.debug(f"{batch_correct_text}, {batch_data}, {batch_source_langs}, {batch_target_langs}")
        #     read_q.put([batch_correct_text, batch_data, batch_source_langs, batch_target_langs])
        #     # self._logger.debug("queueing done")

        # for i in range(num_cores):
        #     read_q.put(None)

        # self._logger.debug("queueing done")
        # # process_pool = mp.Pool(num_cores, initializer=self.process, initargs=(read_q, write_q, iolock,))
        # infer_processes = [mp.Process(target=self.process, args=(read_q, write_q, iolock,)) for _ in range(num_cores)]
        # for process in infer_processes:
        #     self._logger.debug(f"{process} started")
        #     process.start()

        # store_processes = [mp.Process(target=self.store, args=(write_q, iolock,))]
        # for process in store_processes:
        #     self._logger.debug(f"{process} started")
        #     process.start()

        # # Using only 1 process to consume q and write to file
        # # store_pool = mp.Pool(1, initializer=self.store, initargs=(write_q, iolock))

        # self._logger.debug("all processes started")

        # for process in infer_processes:
        #     process.join()

        # for process in store_processes:
        #     process.join()

        # process_pool.close()
        # process_pool.join()

        # store_pool.close()
        # store_pool.join()

        with open(self.config.OUTPUT_FILE, "w") as opf:
            batch_results = []

            # Batch 1 as of now. Can change in future, hence supporting
            for batch_correct_text, batch_data, batch_source_langs, batch_target_langs in tqdm(self.get_inputs()):
                try:
                    # self._logger.debug(f"model: {self.model}")
                    results = self.model.invoke(
                        input=batch_data, source_language=batch_source_langs, target_language=batch_target_langs
                    )["data"]
                    # self._logger.debug(f"result: {results}")
                except Exception as e:
                    raise
                    error_rows.append([os.path.basename(batch_correct_text[0]), batch_data[0]])
                    continue

                batch_results.append(json.dumps({"output": results, "target_sentence": batch_correct_text[0], "target_language": batch_target_langs[0]}) + "\n")
                opf.writelines(batch_results)

        pd.DataFrame(error_rows, columns=["target_sentence", "source_sentence"]).to_csv(
            f"{self.preprocessor_config.__class__.__name__.replace('PreProcessorConfig', '')}_error_files.csv"
        )
        self._logger.info("Total Time Taken {}".format(time.time() - start_time))
        return self.config.OUTPUT_FILE

    def process(self, read_q, write_q, iolock):
        error_rows = []
        while(True):
            self._logger.debug("------ before read_q.get")
            input_data = read_q.get(block=True)
            self._logger.debug(f"----- read {input_data}")
            if input_data is None:
                self._logger("none")
                break

            batch_correct_text, batch_data, batch_source_langs, batch_target_langs = input_data
            try:
                self._logger.debug(f"----- invoke")
                results = self.model.invoke(
                    input=batch_data, source_language=batch_source_langs, target_language=batch_target_langs
                )["data"]
                self._logger.debug(f"result 1: {results}")    
            except Exception as e:
                raise
                self._logger.debug(f"{e}")
                error_rows.append([os.path.basename(batch_correct_text[0]), batch_data[0]])
                continue

            with iolock:
                self._logger.debug(f"result2: {results}")
                # write_q.put(json.dumps({"output": results, "target_sentence": batch_correct_text}) + "\n")
                self._logger.debug(f"write_q put")
            sleep(0.1)

        write_q.put(None)
        with iolock:
            pd.DataFrame(error_rows, columns=["target_sentence", "source_sentence"]).to_csv(
                f"{self.preprocessor_config.__class__.__name__.replace('PreProcessorConfig', '')}_error_files.csv"
            )

    def store(self, write_q, iolock):
        with open(self.config.OUTPUT_FILE, "w") as opf:
            while(True):
                print("------ before write_q.get")
                output = write_q.get(block=True)
                print("write_q.get")
                if output is None:
                    break

                with iolock:
                    opf.writelines(output)
                    # self._logger.debug(f"result: {output}")
                sleep(0.1)


# Dataset specific scorers
# Easy to configure later compared to just having a bunch of if statements

FLORES200NMTScorer = class_factory(
    "FLORES200NMTScorer",
    (NMTScorer,),
    {"config": FLORES200ScorerConfig(), "preprocessor_config": FLORES200PreProcessorConfig()}
)


WAT20NMTScorer = class_factory(
    "WAT20NMTScorer",
    (NMTScorer,),
    {"config": WAT20ScorerConfig(), "preprocessor_config": WAT20PreProcessorConfig()}
)


WAT21NMTScorer = class_factory(
    "WAT21NMTScorer",
    (NMTScorer,),
    {"config": WAT21ScorerConfig(), "preprocessor_config": WAT21PreProcessorConfig()}
)


WMTNMTScorer = class_factory(
    "WMTNMTScorer",
    (NMTScorer,),
    {"config": WMTScorerConfig(), "preprocessor_config": WMTPreProcessorConfig()}
)


PMINMTScorer = class_factory(
    "PMINMTScorer",
    (NMTScorer,),
    {"config": PMIScorerConfig(), "preprocessor_config": PMIPreProcessorConfig()}
)


UFALSNMTScorer = class_factory(
    "UFALSNMTScorer",
    (NMTScorer,),
    {"config": UFALSScorerConfig(), "preprocessor_config": UFALSPreProcessorConfig()}
)
