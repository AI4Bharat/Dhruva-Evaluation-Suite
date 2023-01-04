import logging
from dataclasses import dataclass
from typing import List, Any

import gevent.ssl
import tritonclient.http as http_client
from tritonclient.utils import InferenceServerException

from plugins import PluginBase
from plugins.models.config import ASRBatchConfig


class ASRBatchE2EModel(PluginBase):
    """
    Model class
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Entry init block for plugins
        :param config: options for the plugin
        """

        self.kwargs = kwargs
        self.asr_config = ASRBatchConfig()

        self.headers = {}
        self.headers["Authorization"] = f"Bearer {self.asr_config.API_KEY}"

        self.triton_client = http_client.InferenceServerClient(
            url=self.asr_config.HTTP_URL,
            ssl=True,
            ssl_context_factory=gevent.ssl._create_default_https_context,
        )
        health_ctx = self.triton_client.is_server_ready(headers=self.headers)
        self._logger.info("Is server ready - {}".format(health_ctx))
        # self._logger.debug(self.kwargs)

    # def get_inputs(self, *args, **kwargs):
    #     self.batch_audio_raw = self.kwargs["preprocessed_outputs"]

    def get_inputs(self, audio_raw):
        input0 = http_client.InferInput("WAVPATH", audio_raw.shape, "UINT8")
        input0.set_data_from_numpy(audio_raw)
        output0 = http_client.InferRequestedOutput("TRANSCRIPTS")
        return input0, output0


    def invoke(self, *args, **kwargs) -> Any:
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :param kwargs: possible keyword arguments for the plugin
        :return: None
        """

        input0, output0 = self.get_inputs(kwargs["input"])

        response = self.triton_client.infer(
            self.asr_config.MODEL_NAME,
            model_version=self.asr_config.MODEL_VERSION,
            inputs=[input0],
            request_id=str(1),
            outputs=[output0],
            headers=self.headers
        )
        result_response = response.get_response()
        encoded_result = response.as_numpy("TRANSCRIPTS")
        return [result.decode("utf-8") for result in encoded_result.tolist()]
