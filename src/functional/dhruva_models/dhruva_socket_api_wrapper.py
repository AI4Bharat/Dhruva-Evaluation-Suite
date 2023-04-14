import json
import base64
import logging
from typing import List

import socketio
import datasets
from tqdm import tqdm
from pydub import AudioSegment

BATCH_LEN = 5


feature = datasets.Audio()


def _encode_audio(raw_input):
    data = feature.encode_example(raw_input)
    return base64.b64encode(data["bytes"]).decode("utf-8")


def generate_asr_payload():
    return [
        {
            "task": {"type": "asr"},
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
    print("response: ", response)
    # payload = ULCAAsrInferenceResponse(**response)
    # return [{"text": p.source} for p in payload.output]


def generate_nmt_payload(batch_data: list, input_column: str):
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
        self.is_speaking = False
        self.is_stream_inactive = True

        self.socket_client = self._get_client(
            on_ready=None
        )

        self.socket_client.connect(
            url=socket_url,
            transports=["websocket", "polling"],
            auth={"authorization": api_key},
        )

    def response_handler(self, response):
        output_task = None
        for task in self.task_sequence:
            output_task = task["name"]

        if output_task == "asr":
            self.parsed_response = parse_asr_response(response)
        elif output_task == "nmt":
            self.parsed_response = parse_nmt_response(response)

    def _get_client(self, on_ready=None) -> socketio.Client:
        sio = socketio.Client(reconnection_attempts=5)

        @sio.event
        def connect():
            print("Socket connected with ID:", sio.get_sid())
            sio.emit("start", data=(self.task_sequence))

        @sio.event
        def connect_error(data):
            print("The connection failed!")

        @sio.on("ready")
        def ready():
            self.is_stream_inactive = False
            # print("Server ready to receive data from client")
            if on_ready:
                on_ready()

        @sio.on("response")
        def response(response):
            print(response)
            print()
            if self.is_stream_inactive:
                self.response_handler(response)

        @sio.on("terminate")
        def terminate():
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
        # cut the file into 2 sec chunks amd emit every chunk
        print("audio_data: ", data)
        data = feature.encode_example(data)

        segment = AudioSegment(data["bytes"], sample_width=2, frame_rate=16000, channels=1)
        duration = segment.duration_seconds
        stream_duration = 2

        for i in range(0, duration, stream_duration):
            t1 = i * 1000 # Works in milliseconds
            t2 = i * 1000 + stream_duration * 1000
            chunk = segment[t1:t2]
            print("chunk array: ", chunk)
            chunk = feature.encode_example(chunk)
            print("chunk: ", chunk)

            clear_server_state = not self.is_speaking
            streaming_config = {
                "response_depth": self.task_sequence__intermediate_response_depth
            }
            data = base64.b64encode(chunk["bytes"]).decode("utf-8")
            input_data = {"audio": [{"audioContent": data}]}
            self.socket_client.emit(
                "data",
                data=(
                    input_data,
                    streaming_config,
                    clear_server_state,
                    self.is_stream_inactive,
                ),
            )
        self._transmit_end_of_stream()

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
        try:
            streamer = DhruvaStreamingClient(
                socket_url=self.url,
                api_key=self.api_key,
                task_sequence=globals()[f"generate_{self.task}_task_sequence"](),
                auto_start=False,
            )
            print("sid: ", streamer.socket_client.get_sid())
            streamer.send_file(data)

        except Exception as e:
            import traceback
            print(traceback.format_exc(e))

        return streamer.parsed_response

    def __call__(self, all_audios, **kwargs):
        all_results = []
        for audio in tqdm(all_audios):
            self._infer(audio)
        return all_results
