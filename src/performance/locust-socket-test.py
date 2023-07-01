import time
import gevent
import socketio
from urllib.parse import urlencode

NUM_ALLOWED_HITS = 3


@events.init_command_line_parser.add_listener
def _(parser):
    pass
    # parser.add_argument(
    #     "--service_id",
    #     type=str,
    #     env_var="SERVICE_ID",
    #     default="",
    #     help="Service ID to hit",
    # )
    # parser.add_argument(
    #     "--api_key",
    #     type=str,
    #     env_var="API_KEY",
    #     default="",
    #     help="API key for the service",
    # )
    # parser.add_argument(
    #     "--input_filepath",
    #     type=str,
    #     env_var="INPUT_FILEPATH",
    #     default="",
    #     help="Input file for the service",
    # )

    # Set `include_in_web_ui` to False if you want to hide from the web UI
    # parser.add_argument("--scorer", include_in_web_ui=False, default="I am invisible")
    # Set `is_secret` to True if you want the text input to be password masked in the web UI
    # parser.add_argument("--my-ui-password-argument", is_secret=True, default="I am a secret")


@events.test_start.add_listener
def _(environment, **kw):
    print(f"Custom argument supplied: {environment.parsed_options}")


task_sequence = [
    {
        # "serviceId": "ai4bharat/conformer-en-gpu--t4",
        "taskType": "asr",
        "config": {
            "language": {"sourceLanguage": "hi"},
            "samplingRate": 8000,
            "audioFormat": "pcm",
            "encoding": None,
            # "channel": "mono",
            # "bitsPerSample": "sixteen"
        },
    },
    {
        # "serviceId": "ai4bharat/indictrans-fairseq-all-gpu--t4",
        "taskType": "translation",
        "config": {"language": {"sourceLanguage": "hi", "targetLanguage": "en"}},
    },
    {
        # "serviceId": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
        "taskType": "tts",
        "config": {"language": {"sourceLanguage": "en"}, "gender": "male"},
    },
]


def SocketObj(on_ready=False):
    sio = socketio.Client(reconnection_attempts=5)

    @sio.event
    def connect():
        print("Socket connected with ID:", sio.get_sid())
        # sio.emit("connect_mic_stream")
        sio.emit("start", data=(task_sequence))

    @sio.on("connect-success")
    def ready_to_stream(response=""):
        is_stream_inactive = False
        # print("Server ready to receive data from client")
        if on_ready:
            on_ready()

    @sio.on("response")
    def handle_response(response):
        print("---> response: ", response)

    @sio.on("terminate")
    def terminate(response=""):
        sio.disconnect()

    @sio.event
    def connect_error(data):
        print("The connection failed!")

    @sio.event
    def disconnect():
        print("Stream disconnected!")

    return sio


class SocketIOUser(User):
    streaming_rate: int = 160
    language_code: str = "hi"
    sampling_rate: int = 16000
    post_processors: list = []
    service_id = "ai4bharat/conformer-hi-gpu--t4"
    # api_key: str = "33424379-5c9f-4d6d-8a25-cd56186122b1"
    api_key: str = "f2c3a8c9-aadf-44b0-8b1d-51cfbd16931b"
    is_speaking: bool = True
    is_stream_inactive: bool = False
    clear_server_state: bool = False
    in_data: list = [[]]

    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hits = 1
        input_filepath = "./payloads/2sec_audio.wav"
        with open(input_filepath, "rb") as f:
            self.in_data = f.read()
            self.input_data = {"audio": [{"audioContent": self.in_data}]}

    def init_conn(self):
        self.num_hits = 1
        self.ws = SocketObj()
        self.ws.connect(
            self.socket_url,
            transports=["websocket", "polling"],
            auth={"authorization": self.api_key},
        )
        print("---- init conn ----", self.ws.get_sid())
        self.ws.on("response", self.on_message)
        self.ws.on("disconnect", self.disconnect)

    def connect(self):
        query_string = urlencode(
            {
                "language": self.language_code,
                "samplingRate": self.sampling_rate,
                "postProcessors": self.post_processors,
                "serviceId": self.service_id,
                "apiKey": self.api_key,
            }
        )

        self.socket_url = "wss://api.dhruva.ai4bharat.org"
        # ws_url = "wss://api.dhruva.ai4bharat.org" + "?" + query_string

        try:
            self.init_conn()
            # gevent.spawn(self.receive_loop)
        except Exception as erro:
            # Capture failed events in locust
            self.environment.events.request.fire(
                request_type="WSS Connect",
                name="WSS Connect",
                response_time=0,
                response_length=0,
                exception=str(erro),
                context={"place": "connection"},
            )

    def disconnect(self):
        print("disconnected", self.ws.get_sid())
        self.disconnect_start_time = time.time()

    def on_message(self, message):
        self.result = message
        # print("message: ", self.result)
        print("received", self.ws.get_sid())

        self.environment.events.request.fire(
            request_type="WSR",
            name="WSS Recv",
            response_time=int((time.time() - self.start_at) * 1000),
            response_length=len(message),
            exception=None,
            context=self.context(),
        )
        # print("env fired")

    def receive_loop(self):
        pass
        # this is not needed since in socketio we use events
        # while True:
        # message = self.ws.handle_response()
        # logging.debug(f"WSR: {message}")
        # self.on_message(message)

    def _transmit_end_of_stream(self) -> None:
        # print("end of stream")
        # Convey that speaking has stopped
        self.clear_server_state = True
        self.ws.emit("data", (None, None, self.clear_server_state, self.is_stream_inactive))
        # Convey that we can close the stream safely
        self.is_stream_inactive = True
        self.ws.emit("data", (None, None, self.clear_server_state, self.is_stream_inactive))

        self.clear_server_state = False
        self.is_stream_inactive = False

    def send(self):
        self.start_at = time.time()

        if self.num_hits >= NUM_ALLOWED_HITS:
            print("-------- stopping --------", self.ws.get_sid())
            self._transmit_end_of_stream()
            self.ws.wait()
            self.ws.disconnect()

            self.environment.events.request.fire(
                request_type="WSR",
                name="WSS Stop",
                response_time=int((time.time() - self.disconnect_start_time) * 1000),
                response_length=len(""),
                exception=None,
                context=self.context(),
            )
            # self.sleep_with_heartbeat(2)
            self.START_FLAG = False
            return

        # self.ws.emit("mic_data", data=(self.in_data, self.language_code, self.is_speaking, self.is_stream_inactive))
        streaming_config = {"response_depth": 2}
        # print("in send")
        try:
            self.ws.emit(
                "data",
                data=(
                    self.input_data,
                    streaming_config,
                    self.clear_server_state,
                    self.is_stream_inactive,
                ),
            )
            print("sent", self.ws.get_sid())
            self.num_hits += 1
        except Exception:
            pass

        self.sleep_with_heartbeat(2)

    def sleep_with_heartbeat(self, seconds):
        # print("in sleep")
        while seconds >= 0:
            gevent.sleep(min(15, seconds))
            seconds -= 15
            self.ws.send("2")
        # print("out of sleep")


class MySocketIOUser(SocketIOUser):
    START_FLAG = False
    result = ""

    @task()
    def publish(self):
        self.result = ""
        if not self.START_FLAG:
            self.connect()
            self.START_FLAG = True

        self.send()

        # wait until I get a push message to on_message
        while not self.result:
            time.sleep(0.1)

        # wait for additional pushes, while occasionally sending heartbeats, like a real client would
        # self.sleep_with_heartbeat(2)


if __name__ == "__main__":
    pass

# locust -f locust-socket-test.py --users 2 --spawn-rate 1 -H http://api.dhruva.ai4bharat.org
# locust -f locust_socket.py --users 2 --spawn-r
