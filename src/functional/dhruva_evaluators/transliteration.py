import os
# import json
# import tarfile
# import logging
# from pathlib import Path
# from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal

# from tqdm import tqdm
# import numpy as np
# import soundfile as sf
# from numbers import Number

import multiprocessing as mp
from datasets import Dataset
from evaluate import Evaluator, EvaluationModule

from dhruva_preprocessors import clean_and_normalize_transcripts
# from evaluate.utils.file_utils import add_end_docstrings, add_start_docstrings
# from evaluate.evaluator.base import EVALUATOR_COMPUTE_RETURN_DOCSTRING, EVALUTOR_COMPUTE_START_DOCSTRING


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from dhruva_evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("dhruva-transliteration")
    >>> data = load_dataset("ai4bharat/Aksharantar", "en", split="test[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline=DhruvaModel,
    >>>     data=data,
    >>>     input_column="native_word",
    >>>     label_column="english_word",
    >>>     metric="cer",
    >>> )
    ```
"""

# Figure out how to pass config for datasets, preprocessors, postprocessors and metrics via evaluator
class DhruvaTransliterationEvaluator(Evaluator):
    """
    Dhruva Automatic speech recognition evaluator.
    Methods in this class assume a data format compatible with Dhruva.
    """

    def __init__(self,dataset_name, task="dhruva-transliteration", default_metric_name="cer"):
        super().__init__(task, default_metric_name=default_metric_name)
        self.dataset_name = dataset_name

    def prepare_data(self, data: Dataset, input_column: str, label_column: str, *args, **kwargs):
        """
        Prepare data.
        Args:
            data (`Dataset`): Specifies the dataset we will run evaluation on.
            input_column (`str`, defaults to `"text"`):
                the name of the column containing the text feature in the dataset specified by `data`.
            label_column (`str`, defaults to `"label"`):
                the name of the column containing the labels in the dataset specified by `data`.
        Returns:
            `dict`:  metric inputs.
            `list`:  pipeline inputs.
        """
        self.check_required_columns(data, {"input_column": input_column, "label_column": label_column})
        # preprocess data based on language
        # data = data.map(clean_and_normalize_transcripts, load_from_cache_file=False, disable_nullable=True)  # , num_proc=mp.cpu_count())
        # concatenate_texts is for WER score to be calculated for the whole dataset]
        # data = data.select(list(range(5)))
        return {"references": data[label_column]}, data

    def predictions_processor(self, predictions, label_mapping):
        # print(predictions)
        return {"predictions": [pred["text"] for pred in predictions]}
