import multiprocessing as mp
from datasets import Dataset
from evaluate import Evaluator

from dhruva_preprocessors import clean_and_normalize_transcripts


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from dhruva_evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("dhruva-asr")
    >>> data = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline=DhruvaModel,
    >>>     data=data,
    >>>     input_column="audio",
    >>>     label_column="transcript",
    >>>     metric="wer",
    >>> )
    ```
"""


class DhruvaASREvaluator(Evaluator):
    """
    Dhruva Automatic speech recognition evaluator.
    Methods in this class assume a data format compatible with Dhruva.
    """

    def __init__(
        self,
        dataset_name: str,
        source_language: str,
        target_language: str = None,
        task="dhruva-asr",
        default_metric_name="wer",
    ):
        super().__init__(task, default_metric_name=default_metric_name)
        self.source_language = source_language

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
        data = data.map(
            lambda x: clean_and_normalize_transcripts(x, label_column, self.source_language),
            load_from_cache_file=False,
            disable_nullable=True,
            num_proc=mp.cpu_count(),
        )

        # concatenate_texts is for WER score to be calculated for the whole dataset
        return {"references": data[label_column], "concatenate_texts": True}, data

    def predictions_processor(self, predictions, label_mapping):
        return {"predictions": [pred["text"] for pred in predictions]}
