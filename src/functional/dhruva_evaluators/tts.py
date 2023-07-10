import os


import multiprocessing as mp
from datasets import Dataset
from evaluate import Evaluator, EvaluationModule

TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from dhruva_evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("dhruva-tts")
    >>> data = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline=DhruvaModel,
    >>>     data=data,
    >>>     input_column="transcript",
    >>>     label_column="audio",
    >>>     metric="mcd",
    >>> )
    ```
"""


# Figure out how to pass config for datasets, preprocessors, postprocessors and metrics via evaluator
class DhruvaTTSEvaluator(Evaluator):
    """
    Dhruva Text to speech evaluator.
    Methods in this class assume a data format compatible with Dhruva.
    """

    def __init__(
        self,
        dataset_name,
        source_language: str,
        target_language: str,
        task="dhruva-tts",
        default_metric_name="mcd/evaluate_mcd.py",
    ):
        super().__init__(task, default_metric_name=default_metric_name)
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
        return {"predictions": [pred["path"] for pred in predictions]}
