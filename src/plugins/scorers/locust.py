import gevent.monkey
gevent.monkey.patch_all()

import os
import sys
import copy
import time
import json
import base64
import logging
from glob import glob
from typing import Any, List, Union, Iterable, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
# import tritonclient.grpc as grpc_client
import tritonclient.http as http_client
from tritonclient.utils import InferenceServerException

from plugins import PluginBase
from config import BaseConfig
from plugins.scorers.asr_batch_off_conf import MUCSBatchOffConfScorer
# from plugins.preprocessors.config import MUCSHindiConfig as MUCSHindiPPConfig
from plugins.scorers.config import MUCSScorerConfig


import gevent
import gevent.ssl
from locust import HttpUser, User, task, constant, constant_throughput, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging


class LocustConfig():
    PARALLELISM: int = 10
    TEST_DURATION: int = 100  # (secs)
    SPAWN_RATE: int = 1


class ASRUser(HttpUser):
    wait_time = between(0, 0)
    host = "http://127.0.0.1:8001"

    def on_start(self):
        print(f"Starting {self.__class__}")
        for _, _, batch_audio_raw in self.environment.scorer.get_inputs():
            self.environment.data = batch_audio_raw
            break

    @task
    def inference(self):
        model_inputs = self.environment.model.get_inputs(self.environment.data)
        headers, payload = self.environment.model.generate_ULCA_payload(
            model_inputs, self.environment.model.config.MODEL_TYPE
        )
        response = self.environment.model.infer(
            self.environment.model.config.HTTP_URL, headers, payload, client=self.client
        )


class LocustScorer(PluginBase):
    def __init__(self, *args, **kwargs):
        self.env = Environment()
        self.runner = self.env.create_local_runner()

        # start a WebUI instance
        self.web_ui = self.env.create_web_ui("127.0.0.1", 8089)
        self._logger.info("starting scorer")

    def invoke(self, *args, **kwargs):
        print("starting invokation")
        # execute init event handlers (only really needed if you have registered any)
        self.env.events.init.fire(environment=self.env, runner=self.runner, web_ui=self.web_ui)

        # start a greenlet that periodically outputs the current stats
        gevent.spawn(stats_printer(self.env.stats))

        # start a greenlet that save current stats to history
        gevent.spawn(stats_history, self.runner)

        # start the test
        self.runner.start(self.locust_config.PARALLELISM, spawn_rate=self.locust_config.SPAWN_RATE)

        # in 60 seconds stop the runner
        gevent.spawn_later(self.locust_config.TEST_DURATION, lambda: self.runner.quit())

        # wait for the greenlets
        self.runner.greenlet.join()
        # stop the web server for good measures
        self.web_ui.stop()
        self._logger.info("Completed the load test")


class LocustASRScorer(LocustScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = Environment(user_classes=[ASRUser])
        self.locust_config = LocustConfig()

        self.runner = self.env.create_local_runner()
        # start a WebUI instance
        self.web_ui = self.env.create_web_ui("127.0.0.1", 8088)

        # Dynamically init the scorer based on config
        self.env.scorer = MUCSBatchOffConfScorer(**kwargs)
        self.input = []
        self.env.model = self.env.scorer.model


if __name__ == "__main__":
    ls = LocustScorer()
    ls.invoke()



# Context Manager / dependency in FastAPI
# Figure out conformer model
# Locust dependency for data and client

# Eval
# init gets locust client
# get inputs
# convert to ULCA
# call via locust or normal client

# Locust
# on_start inits and gets input
# converts into ULCA
# calls invoke for the infer method
# No need to sub class or extend the Locust HTTP client
