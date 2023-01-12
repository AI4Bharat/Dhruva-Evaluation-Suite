import logging
from dataclasses import dataclass
from typing import List, Any
import base64
import gevent.ssl
import tritonclient.http as http_client
from tritonclient.utils import InferenceServerException
import requests
from plugins import PluginBase
from plugins.models.config import ASRBatchConfig, IndicTinyASRConfig
import json

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


class IndicTinyASRModel(PluginBase):
    def __init__(self, *args, **kwargs) -> None:
        """
        Entry init block for plugins
        :param config: options for the plugin
        """

        self.kwargs = kwargs
        self.asr_config = IndicTinyASRConfig()

        self.headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Origin": "https://models.ai4bharat.org",
            "Content-Length": "147503",
            "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
            "Host": "asr-api.ai4bharat.org",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
            "Referer": "https://models.ai4bharat.org/",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def invoke(self, *args, **kwargs) -> Any:
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :param kwargs: possible keyword arguments for the plugin
        :return: None
        """

        #input0, output0 = self.get_inputs(kwargs["input"])
        wav_file = kwargs["input"].tobytes()
        encoded_string = base64.b64encode(wav_file)
        #Encode the file.
        encoded_string = str(encoded_string,'ascii','ignore')
        
        # POST request data format
        payload = {
            "config": {
                "language": {
                    "sourceLanguage": "hi",
                },
                "transcriptionFormat": {
                    "value": "transcript",
                },
                "audioFormat": "wav",
                "samplingRate": "16000",
                "postProcessors": None
            },
            "audio": [{"audioContent": encoded_string}],
        }

       
        resp = requests.post(self.asr_config.HTTP_URL, headers=self.headers, data=json.dumps(payload))
        print(json.loads(resp.text)["output"][0]["source"])
        return json.loads(resp.text)["output"][0]["source"]
        # result_response = response.get_response()
        # encoded_result = response.as_numpy("TRANSCRIPTS")
        # return [result.decode("utf-8") for result in encoded_result.tolist()]






