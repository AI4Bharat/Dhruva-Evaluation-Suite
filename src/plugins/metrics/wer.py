import os
import json
# from pathlib import Path
from typing import List, Any

from jiwer import wer
from plugins import PluginBase
# from plugins.metrics.config import WERConfig
from plugins.scorers.config import ASRBatchInputValidation


class WERMetric(PluginBase):
    """
    - Get datasets from any URL
    """

    def __init__(self, **kwargs) -> None:
        """
        :param config: options for the plugin
        """
        self.kwargs = kwargs
        self.scorer_config = ASRBatchInputValidation()
        # self.config = None

    def invoke(self, *args, **kwargs) -> Any:
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :param kwargs: possible keyword arguments for the plugin
        :return: None
        """

        if not os.path.exists(self.scorer_config.OUTPUT_FILE):
            self._logger.error("No output file! SKipping metric calculation...")
            return False

        all_data = []
        with open(self.scorer_config.OUTPUT_FILE, "r") as ipf:
            for line in ipf:
                all_data.append(json.loads(line))

        hypothesis = []
        ground_truth = []
        for d in all_data:
            hypothesis.append(d["output"])
            ground_truth.append(d["transcript"])

        error = wer(ground_truth, hypothesis)
        self._logger.info(f"WER is: {error}")
