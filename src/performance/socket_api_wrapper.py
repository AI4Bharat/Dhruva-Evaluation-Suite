import gevent
import socketio
import numpy as np
from urllib.parse import urlencode
from scipy.io.wavfile import write, read
from locust import User, task, between, events

NUM_ALLOWED_HITS = 3


task_sequence = [
    {
        "taskType": "asr",
        "config": {
            "language": {"sourceLanguage": "hi"},
            "samplingRate": 16000,
            "audioFormat": "wav",
            "encoding": "base64",
            # "channel": "mono",
            # "bitsPerSample": "sixteen"
        },
    },
    {
        "taskType": "translation",
        "config": {"language": {"sourceLanguage": "hi", "targetLanguage": "en"}},
    },
]


class SocketIOUser(User):
    api_key: str = "99685ac6-1b71-4064-aa01-c0b2fbbb792e"
    socket_url: str = "wss://dhruva-api.bhashini.gov.in"
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.is_speaking = True
        self.is_stream_inactive = False

    def connect(self):
        # states
        self.is_speaking = True
        self.is_stream_inactive = False
        self.socket_client = self._get_client(on_ready=False)
        try:
            self.socket_client.connect(
                url=self.socket_url,
                transports=["websocket", "polling"],
                auth={"authorization": self.api_key},
            )
            # self.streamer.socket_client.on("response", self.on_message)
            self.socket_client.on("disconnect", self.disconnect)
            self.socket_client.on("response", self.response_handler)
            print("---- init conn ----", self.socket_client.get_sid())

        except Exception as erro:
            # Capture failed events in locust
            self.environment.events.request.fire(
                request_type="WSS Connect",
                name="Connect",
                response_time=0,
                response_length=0,
                exception=str(erro),
                context={"place": "connection"},
            )

    def response_handler(self, response, response_type):
        print("response")
        self.result = response

        self.environment.events.request.fire(
            request_type="WSR",
            name="Response",
            response_time=int((time.time() - self.start_at) * 1000),
            response_length=len(response),
            exception=None,
            context=self.context(),
        )

        if self.is_stream_inactive:
            self.environment.events.request.fire(
                request_type="WSR",
                name="Response Complete",
                response_time=int((time.time() - self.start_at) * 1000),
                response_length=len(response),
                exception=None,
                context=self.context(),
            )

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
        def response(response, response_type):
            print("--> response 1: ")
            if self.is_stream_inactive:
                self.response_handler(response, response_type)

        @sio.on("terminate")
        def terminate():
            print("terminate")
            sio.disconnect()

        @sio.event
        def disconnect():
            print("Stream disconnected!")

        return sio

    def send_file(self):
        stream_duration = 2

        input_filepath = "0116_025.wav"
        sr, sound = read(input_filepath)
        # np.array(sound, dtype=float)

        self.start_at = time.time()
        self.disconnected = False
        slices = np.arange(0, len(sound) / sr, stream_duration, dtype=np.int32)

        for j, (start, end) in enumerate(zip(slices[:-1], slices[1:])):
            start_audio = start * sr
            end_audio = end * sr
            audio_slice = sound[int(start_audio) : int(end_audio)]

            clear_server_state = not self.is_speaking
            streaming_config = {
                "response_depth": self.task_sequence__intermediate_response_depth
            }
            bytes_wav = bytes()
            byte_io = io.BytesIO(bytes_wav)
            write(byte_io, sr, audio_slice)
            result_bytes = byte_io.read()

            input_data = {"audio": [{"audioContent": base64.b64encode(result_bytes)}]}
            print(f"Step {j}")
            self.socket_client.emit(
                "data",
                data=(
                    input_data,
                    streaming_config,
                    clear_server_state,
                    self.is_stream_inactive,
                ),
            )
            time.sleep(2)

        self.disconnect_start_time = time.time()
        self.stop()

    def _transmit_end_of_stream(self) -> None:
        # Convey that speaking has stopped
        clear_server_state = not self.is_speaking
        # clear_server_state = True
        self.socket_client.emit(
            "data", (None, None, clear_server_state, self.is_stream_inactive)
        )
        # Convey that we can close the stream safely
        self.is_stream_inactive = True
        self.socket_client.emit(
            "data", (None, None, clear_server_state, self.is_stream_inactive)
        )
        print("Terminated")

    def stop(self, force=False) -> None:
        print("Stopping...")
        self.is_speaking = False
        self._transmit_end_of_stream()

        # Wait till stream is disconnected
        self.socket_client.wait()

    def force_disconnect(self, sig=None, frame=None) -> None:
        self.socket_client.disconnect()

    def disconnect(self):
        print("disconnected", self.socket_client.get_sid())
        self.environment.events.request.fire(
            request_type="WSR",
            name="Stop",
            response_time=int((time.time() - self.disconnect_start_time) * 1000),
            response_length=len(""),
            exception=None,
            context=self.context(),
        )
        self.disconnected = True

    def sleep_with_heartbeat(self, seconds):
        # print("in sleep")
        while seconds >= 0:
            gevent.sleep(min(15, seconds))
            seconds -= 15
            self.ws.send("1")
        # print("out of sleep")


class MySocketIOUser(SocketIOUser):
    @task()
    def publish(self):
        self.result = ""
        self.connect()
        self.send_file()

        while not self.disconnected:
            time.sleep(0.1)


if __name__ == "__main__":
    pass
