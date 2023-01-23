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
import plugins.datasets.config as dataset_config


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

        self._logger.info("downloading dataset...")
        self.download(self.config.DATASET_URL, self.config.LOCAL_PATH)
        self._logger.info("finished downloading dataset...")
        return self.config.LOCAL_PATH

    # Add support for showing bytes / total bytes downloaded
    def download(self, url, local_path):
        get_response = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in tqdm(get_response.iter_content(chunk_size=1024*10)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)


# ToDo
# Downlaod all versions - splits - langs of a dataset within a single dataset class
# Going for one class per version - lang - split as of now


# ASR
IndicSUPERBKnownDataset = class_factory("IndicSUPERBKnownDataset", (DatasetBase,), {"config": dataset_config.IndicSUPERBTestKnownConfig()})
IndicSUPERBUnknownDataset = class_factory("IndicSUPERBUnknownDataset", (DatasetBase,), {"config": dataset_config.IndicSUPERBTestUnknownConfig()})
CommonVoiceDataset = class_factory("CommonVoiceDataset", (DatasetBase,), {"config": dataset_config.CommonVoiceConfig()})
MUCSHindiDataset = class_factory("MUCSHindiDataset", (DatasetBase,), {"config": dataset_config.MUCSHindiConfig()})

# NMT
FLORES200Dataset = class_factory("FLORESDataset", (DatasetBase,), {"config": dataset_config.FLORESDatasetConfig()})
WAT21Dataset = class_factory("WAT21Dataset", (DatasetBase,), {"config": dataset_config.WAT21DatasetConfig()})
WAT20Dataset = class_factory("WAT20Dataset", (DatasetBase,), {"config": dataset_config.WAT20DatasetConfig()})
WMTDataset = class_factory("WMTDataset", (DatasetBase,), {"config": dataset_config.WMTDatasetConfig()})
UFALDataset = class_factory("UFALDataset", (DatasetBase,), {"config": dataset_config.UFALDatasetConfig()})
PMIDataset = class_factory("PMIDataset", (DatasetBase,), {"config": dataset_config.PMIDatasetConfig()})
