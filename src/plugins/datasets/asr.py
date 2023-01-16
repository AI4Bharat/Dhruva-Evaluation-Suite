import os
import logging
import requests
import urllib.request
from pathlib import Path
from typing import List, Any

from tqdm import tqdm
from plugins import PluginBase
from helpers import class_factory
from config import BaseConfig
from plugins.datasets.config import (
    IndicSUPERBTestKnownConfig, IndicSUPERBTestUnknownConfig, MUCSHindiConfig, CommonVoiceConfig
)


class DatasetBase(PluginBase):
    """
    - Get datasets from any URL
    """

    def __init__(self, **kwargs) -> None:
        """
        :param config: options for the plugin
        """
        self.kwargs = kwargs
        # self.config = None

    def invoke(self, *args, **kwargs) -> Any:
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :param kwargs: possible keyword arguments for the plugin
        :return: None
        """
        
        # urllib.request.urlretrieve(IndicSUPERBTestKnownConfig.DATASET_URL, IndicSUPERBTestKnownConfig.LOCAL_PATH)
        # check for presence in buckets
        if os.path.exists(self.config.LOCAL_PATH):
            self._logger.info("Path exists! SKipping download...")
            return self.config.LOCAL_PATH

        self.download(self.config.DATASET_URL, self.config.LOCAL_PATH)
        return self.config.LOCAL_PATH

    # Add support for showing bytes / total bytes downloaded
    def download(self, url, local_path):
        get_response = requests.get(url,stream=True)
        with open(local_path, 'wb') as f:
            for chunk in tqdm(get_response.iter_content(chunk_size=1024*10)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)


# ToDo
# Downlaod all versions - splits - langs of a dataset within a single dataset class
# Going for one class per version - lang - split as of now


IndicSUPERBKnownDataset = class_factory("IndicSUPERBKnownDataset", (DatasetBase,), {"config": IndicSUPERBTestKnownConfig()})
IndicSUPERBUnknownDataset = class_factory("IndicSUPERBUnknownDataset", (DatasetBase,), {"config": IndicSUPERBTestUnknownConfig()})
CommonVoiceDataset = class_factory("CommonVoiceDataset", (DatasetBase,), {"config": CommonVoiceConfig()})
MUCSHindiDataset = class_factory("MUCSHindiDataset", (DatasetBase,), {"config": MUCSHindiConfig()})
