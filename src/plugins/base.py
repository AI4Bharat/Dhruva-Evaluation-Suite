import os
from abc import ABC
import logging
from dataclasses import dataclass
from typing import List, Any, Dict, Optional, Tuple

from config import BaseConfig


class PluginRegistry(type):
    plugin_registry: Dict = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(cls)
        cls._logger = logging.getLogger(BaseConfig.LOGGER)
        if name != 'PluginBase':
            # print("inside meta: ", name)
            cls._logger.info(f"Creating plugin: {name}")
            PluginRegistry.plugin_registry[name] = cls


class PluginBase(object, metaclass=PluginRegistry):
    """
    Plugin core class
    """

    # meta: Optional[Meta]
    # plugins = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # cls.plugins.append(cls)


    def __init__(self, **kwargs) -> None:
        """
        Entry init block for plugins
        :param config: options for the plugin
        """
        # self.config = PluginConfig()
        pass

    def invoke(self, *args, **kwargs) -> Any:
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :param kwargs: possible keyword arguments for the plugin
        :return: None
        """
        raise NotImplementedError

    def get_inputs(self, *args, **kwargs):
        raise NotImplementedError

    # def create_request(self, *args, **kwargs):
    #     raise NotImplementedError
