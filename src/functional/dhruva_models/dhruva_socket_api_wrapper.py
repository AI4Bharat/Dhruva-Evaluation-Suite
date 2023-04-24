import time
import logging

import socketio
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

BATCH_LEN = 5
feature = datasets.Audio()


def generate_asr_task_sequence():
    return [
        {
            "taskType": "asr",
            "config": {
                "language": {"sourceLanguage": "en"},
                "samplingRate": 16000,
                "audioFormat": "wav",
                "encoding": "base64",
                # "channel": "mono",
                # "bitsPerSample": "sixteen"
            }
        }
    ]


def parse_asr_response(response: dict):
    return [{"text": p["source"]} for resp in response["pipelineResponse"] for p in resp["output"]]


def generate_nmt_task_sequence(batch_data: list, input_column: str):
    payload = {
        "config": {
            "language": {
                "sourceLanguage": batch_data[0]["source_language"],
                "sourceScriptCode": "",
                "targetLanguage": batch_data[0]["target_language"],
                "targetScriptCode": "",
            },
            "postProcessors": [],
        }
    }
    payload["input"] = [{"source": data[input_column]} for data in batch_data]
    # payload = ULCATranslationInferenceRequest(**payload)
    return payload.dict()


def parse_nmt_response(response: dict):
    pass
    # payload = ULCATranslationInferenceResponse(**response)
    # return [{"text": p.target} for p in payload.output]


class DhruvaStreamingClient:
    def __init__(
        self,
        socket_url: str,
        api_key: str,
        task_sequence: list,
        auto_start: bool = False,
    ) -> None:
        # Default ASR settings
        self.input_audio__streaming_rate = 640
        self.input_audio__bytes_per_sample = 2
        self.input_audio__sampling_rate = task_sequence[0]["config"]["samplingRate"]
        self.input_audio__num_channels = 1

        # Parameters
        self.task_sequence = task_sequence
        self.task_sequence__depth = len(task_sequence)
        self.task_sequence__intermediate_response_depth = 2  # ASR+NMT

        # states
        self.audio_stream = None
        self.is_speaking = True
        self.is_stream_inactive = False

        self.socket_client = self._get_client(
            on_ready=False
        )

        self.socket_client.connect(
            url=socket_url,
            transports=["websocket", "polling"],
            auth={"authorization": api_key},
        )

    def response_handler(self, response):
        task = ""
        for t in self.task_sequence:
            task = t["taskType"]

        self.parsed_response = globals()[f"parse_{task}_response"](response)
        print("\n\n------\nfinal response: ", self.parsed_response, "\n----------\n")

    def _get_client(self, on_ready=None) -> socketio.Client:
        sio = socketio.Client(reconnection_attempts=5)

        @sio.event
        def connect():
            print("Socket connected with ID:", sio.get_sid(), self.task_sequence)
            sio.emit("start", data=(self.task_sequence))

        @sio.event
        def connect_error(data):
            print("The connection failed!")

        # @sio.on("ready")
        @sio.on("connect-success")
        def ready():
            self.is_stream_inactive = False
            print("Server ready to receive data from client")
            if on_ready:
                on_ready()

        @sio.on("response")
        def response(response):
            # print("response: ", response)
            if self.is_stream_inactive:
                self.response_handler(response)

        @sio.on("terminate")
        def terminate():
            print("terminate")
            sio.disconnect()

        @sio.event
        def disconnect():
            print("Stream disconnected!")

        return sio

    def stop(self) -> None:
        print("Stopping...")
        self.is_speaking = False
        self._transmit_end_of_stream()

        # Wait till stream is disconnected
        self.socket_client.wait()

    def force_disconnect(self, sig=None, frame=None) -> None:
        self.socket_client.disconnect()

    def send_nmt_payload(self, data):
        pass

    def send_file(self, data):
        stream_duration = 2
        frequency = 16000

        slices = np.arange(0, len(data["audio"]["array"])/16000, stream_duration, dtype=np.int)

        for start, end in zip(slices[:-1], slices[1:]):
            start_audio = start * frequency
            end_audio = end * frequency
            audio_slice = data["audio"]["array"][int(start_audio): int(end_audio)]
            chunk = feature.encode_example({
                "array": audio_slice, "path": data["audio"]["path"], "sampling_rate": frequency})

            clear_server_state = not self.is_speaking
            streaming_config = {
                "response_depth": self.task_sequence__intermediate_response_depth
            }
            input_data = {"audio": [{"audioContent": chunk["bytes"]}]}

            self.socket_client.emit(
                "data",
                data=(
                    input_data,
                    streaming_config,
                    clear_server_state,
                    self.is_stream_inactive,
                ),
            )
            # print("before wait")
            # self.socket_client.wait()
            # print("after wait")
            time.sleep(2)

        self._transmit_end_of_stream()
        time.sleep(2)

    def _transmit_end_of_stream(self) -> None:
        # Convey that speaking has stopped
        clear_server_state = not self.is_speaking
        self.socket_client.emit(
            "data", (None, None, clear_server_state, self.is_stream_inactive)
        )
        # Convey that we can close the stream safely
        self.is_stream_inactive = True
        self.socket_client.emit(
            "data", (None, None, clear_server_state, self.is_stream_inactive)
        )
        print("Terminated")


class DhruvaSocketModel:
    def __init__(
        self,
        task: str,
        url: str,
        input_column: str,
        api_key: str,
        **kwargs,
    ):
        self.task = task
        self.url = url
        self.input_language_column = kwargs.get("input_language_column")
        self.input_column = input_column
        self.api_key = api_key

    def _infer(self, data):
        streamer = DhruvaStreamingClient(
            socket_url=self.url,
            api_key=self.api_key,
            task_sequence=globals()[f"generate_{self.task}_task_sequence"](),
            auto_start=False,
        )
        streamer.send_file(data)
        while True:
            if not hasattr(streamer, "parsed_response"):
                time.sleep(1)
                continue
            break

        return streamer.parsed_response

    def __call__(self, all_audios, **kwargs):
        all_results = []
        errors = []
        for audio in tqdm(all_audios):
            try:
                self._infer(audio)
            except Exception:
                errors.append(audio["audio"]["path"])

        pd.DataFrame(errors).to_csv("errors.csv")
        return all_results
