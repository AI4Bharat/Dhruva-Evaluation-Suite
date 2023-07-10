from typing import Union
import multiprocessing as mp
from datasets import Dataset, load_dataset
from evaluate import Evaluator, EvaluationModule, TranslationEvaluator

from constants import Enums, DATASET_INPUT_COLUMN_MAPPING
from dhruva_preprocessors import normalize_language_codes


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from dhruva_evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("dhruva-mt")
    >>> data = load_dataset("facebook/flores200", "en", split="validation[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline=DhruvaModel,
    >>>     data=data,
    >>>     input_column="text",
    >>>     label_column="translation",
    >>>     metric="bleu",
    >>> )
    ```
"""


class DhruvaMTEvaluator(TranslationEvaluator):
    """
    Dhruva Translation evaluator.
    Methods in this class assume a data format compatible with Dhruva.
    """

    def __init__(
        self,
        dataset_name,
        source_language: str,
        target_language: str,
        task=Enums.tasks.NMT,
        default_metric_name="sacrebleu",
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
                the name of the column containing the text feature in the dataset
                specified by `data`.
            label_column (`str`, defaults to `"label"`):
                the name of the column containing the labels in the dataset specified by `data`.
        Returns:
            `dict`:  metric inputs.
            `list`:  pipeline inputs.
        """

        self.check_required_columns(data, {"input_column": input_column, "label_column": label_column})

        # preprocess FLORES data based on language
        if self.dataset_name == Enums.datasets.FLORES:
            source_language = input_column.replace(DATASET_INPUT_COLUMN_MAPPING.get(self.dataset_name), "")
            target_language = label_column.replace(DATASET_INPUT_COLUMN_MAPPING.get(self.dataset_name), "")
            data = data.map(
                normalize_language_codes,
                fn_kwargs={
                    "source_language": source_language,
                    "target_language": target_language,
                },
                num_proc=mp.cpu_count(),
            )

        return {"references": data[label_column]}, data

    def predictions_processor(self, predictions, label_mapping):
        return {"predictions": [pred["text"] for pred in predictions]}
