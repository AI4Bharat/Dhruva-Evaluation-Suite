# This is required
import gevent.monkey
gevent.monkey.patch_all()


import os
import logging
from abc import ABC
from typing import Dict, Tuple, List
from dataclasses import dataclass

from pathlib import Path
from importlib import util

import helpers
from config import BaseConfig
from plugins import PluginRegistry


@dataclass
class PluginConfig:
    name: str
    alias: str
    process_type: str
    description: str
    version: str
    # requirements: Optional[List[DependencyModule]]

    PLUGIN_TYPES: Tuple = (
        "datasets",
        "metrics",
        "preprocessors",
        "postprocessors",
        "models",
        "scorers",
        "metrics"
    )
    PLUGIN_ROOT_PATH: Path = Path("./plugins/")


# Loads modules from a given path
def load_module(path):
    name = os.path.split(path)[-1]
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Utility to load modules within the plugins folders
def load_modules_in_path():
    for ptype in PluginConfig.PLUGIN_TYPES:
        for fname in os.listdir(os.path.join(PluginConfig.PLUGIN_ROOT_PATH, ptype)):
            # Load only proper modules
            if not fname.startswith('.') and not fname.startswith('__') and fname.endswith('.py') and not ("config" in fname):
                try:
                    load_module(os.path.join(os.path.join(PluginConfig.PLUGIN_ROOT_PATH, ptype), fname))
                except Exception:
                    raise


def get_all_plugins() -> List:
    return [name for name, _ in PluginRegistry.plugin_registry.items()]


class EvaluatorBase():
    _plugins = []

    def __init__(self, config: Dict) -> None:
        load_modules_in_path()
        self.plugin_registry = get_all_plugins()
        self.config = config
        self.add_steps()
        self._logger = logging.getLogger(BaseConfig.LOGGER)
        # self._logger.info(f"self.plugin_registry: {self.plugin_registry}")

    def add_steps(self, *args, **kwargs) -> None:
        # Add complex ordering or such requirements in the future
        self._plugins = {
            name: {
                "plugin": PluginRegistry.plugin_registry[name],
                "config": config
            } for name, config in self.config["plugins"].items()
        }

    def eval(self) -> None:
        raise NotImplementedError


class AccuracyEvaluator(EvaluatorBase):

    def eval(self) -> None:
        prev_output = None
        for name, plugin in self._plugins.items():
            name = name.lower()
            self._logger.info(f"In {name}")
            self.config["user_config"] = plugin["config"]
            p = plugin["plugin"](**self.config)

            if "model" in name.lower():
                self.config["model"] = plugin["plugin"]
                continue
            elif "postprocessor" in name.lower():
                self.config["postprocessor"] = plugin["plugin"]
                continue
            # elif "metric" in name.lower():
            #     self.config["metrics"] = plugin["plugin"]
            #     continue

            # ToDo
            # Code to enforce order of modules

            stage_output = p.invoke()
            if "dataset" in name:
                self.config["dataset_output"] = stage_output
            elif "preprocessor" in name:
                self.config["preprocessed_output"] = stage_output
            elif "scorer" in name:
                self.config["scorer_output"] = stage_output
            else:
                pass

        self._logger.info(f"Completed run")


if __name__ == "__main__":
    eval = AccuracyEvaluator(
        {
            "plugins": {
                # "IndicSUPERBKnownDataset": {},
                # "IndicSUPERBUnknownDataset": {},
                "MUCSHindiDataset": {},
                # "CommonVoiceDataset": {},
                # "ASRPreProcessor": {},
                "MUCSPreProcessor": {},
                # "IndicSUPERBKnownPreProcessor": {},
                # "IndicSUPERBUnknownPreProcessor": {},
                # "CommonVoicePreProcessor": {},
                # "ASRBatchE2EModel": {},
                "ASRBatchOffConfModel": {},
                "MUCSBatchOffConfScorer": {},
                # "LocustASRScorer": {}
                # "IndicSUPERBBatchOffConfScorer": {},
                # "MUCSBatchE2EScorer": {}
                # "CommonVoiceBatchE2EScorer": {},
                # "WERMetric": {}

                # NMT
                # "WAT20Dataset": {},
                # "WAT20PreProcessor": {
                #     "preprocessor_config": {
                #         "MODE": "indic-en"
                #     }
                # },
                # "NMTModel": {},
                # "WAT20NMTScorer": {},
                # "BLEUMetric": {},

                # "WMTDataset": {},
                # "WMTPreProcessor": {
                #     "preprocessor_config": {
                #         "MODE": "indic-en"
                #     }
                # },
                # "NMTModel": {},
                # "WMTNMTScorer": {},
                # "BLEUMetric": {},
            }
        }
    )
    eval.eval()