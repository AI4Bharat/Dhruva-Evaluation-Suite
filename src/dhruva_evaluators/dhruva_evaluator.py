import os
import json
import tarfile
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal

from tqdm import tqdm
import numpy as np
import soundfile as sf
from datasets import Dataset
from evaluate import Evaluator, EvaluationModule
from evaluate.utils.file_utils import add_end_docstrings, add_start_docstrings
from evaluate.evaluator.base import EVALUATOR_COMPUTE_RETURN_DOCSTRING, EVALUTOR_COMPUTE_START_DOCSTRING


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from dhruva_evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("automatic-speech-recognition")
    >>> data = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline="https://huggingface.co/openai/whisper-tiny.en",
    >>>     data=data,
    >>>     input_column="path",
    >>>     label_column="sentence",
    >>>     metric="wer",
    >>> )
    ```
"""

# Figure out how to pass config for datasets, preprocessors, postprocessors and metrics via evaluator
class DhruvaASREvaluator(Evaluator):
    """
    Automatic speech recognition evaluator.
    This ASR evaluator can be loaded from [`evaluator`] using the default task name
    `dhruva-automatic-speech-recognition`.
    Methods in this class assume a data format compatible with Dhruva.
    """

    PIPELINE_KWARGS = {"truncation": True}

    def __init__(self, task="dhruva-asr", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def predictions_processor(self, predictions, label_mapping):
        return {"predictions": [pred["text"] for pred in predictions]}

    @add_start_docstrings(EVALUTOR_COMPUTE_START_DOCSTRING)
    @add_end_docstrings(EVALUATOR_COMPUTE_RETURN_DOCSTRING, TASK_DOCUMENTATION)
    def compute(
        self,
        model_or_pipeline: Union[
            str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"
        ] = None,
        data: Union[str, Dataset] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        input_column: str = "path",
        label_column: str = "sentence",
        generation_kwargs: dict = None,
    ) -> Tuple[Dict[str, float], Any]:
        """
        input_column (`str`, defaults to `"path"`):
            the name of the column containing the input audio path in the dataset specified by `data`.
        label_column (`str`, defaults to `"sentence"`):
            the name of the column containing the labels in the dataset specified by `data`.
        generation_kwargs (`Dict`, *optional*, defaults to `None`):
            The generation kwargs are passed to the pipeline and set the text generation strategy.
        """

        if generation_kwargs is not None:
            self.PIPELINE_KWARGS.update(generation_kwargs)

        result = super().compute(
            model_or_pipeline=model_or_pipeline,
            data=data,
            subset=subset,
            split=split,
            metric=metric,
            tokenizer=tokenizer,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            device=device,
            random_state=random_state,
            input_column=input_column,
            label_column=label_column,
        )

        return result
