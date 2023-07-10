import os

import multiprocessing as mp
from datasets import Dataset
from evaluate import Evaluator


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

    def __init__(
        self,
        dataset_name,
        source_language: str,
        target_language: str,
        task="dhruva-transliteration",
        default_metric_name="cer",
    ):
        super().__init__(task, default_metric_name=default_metric_name)
        self.dataset_name = dataset_name
        self.source_language = source_language
        self.target_language = target_language

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
        return {"references": data[label_column]}, data

    def predictions_processor(self, predictions, label_mapping):
        return {"predictions": [pred["text"] for pred in predictions]}
