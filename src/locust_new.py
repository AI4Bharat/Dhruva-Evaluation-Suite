# import gevent
# import gevent.ssl

from locust import HttpUser, User, task, constant, constant_throughput, between, events
# from locust.env import Environment
# from locust.stats import stats_printer, stats_history
# from locust.log import setup_logging

from plugins import PluginRegistry
from evaluator import load_modules_in_path


class LocustConfig():
    PARALLELISM: int = 40
    TEST_DURATION: int = 100  # (secs)
    SPAWN_RATE: int = 1


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--model", type=str, env_var="LOCUST_MODEL", default="", help="Model to hit")
    parser.add_argument("--scorer", type=str, env_var="LOCUST_SCORER", default="", help="Scorer file to use")

    # Set `include_in_web_ui` to False if you want to hide from the web UI
    # parser.add_argument("--scorer", include_in_web_ui=False, default="I am invisible")
    # Set `is_secret` to True if you want the text input to be password masked in the web UI
    # parser.add_argument("--my-ui-password-argument", is_secret=True, default="I am a secret")


@events.test_start.add_listener
def _(environment, **kw):
    print(f"Custom argument supplied: {environment.parsed_options.model}")


class ASRUser(HttpUser):
    wait_time = between(0.01, 0.01)
    # host = "http://host.docker.internal:8001"

    def on_start(self):
        load_modules_in_path()
        model = PluginRegistry.plugin_registry[self.environment.parsed_options.model]
        self.environment.scorer = PluginRegistry.plugin_registry[self.environment.parsed_options.scorer](model=model)

        for _, _, batch_audio_raw in self.environment.scorer.get_inputs():
            self.environment.data = batch_audio_raw
            break

    @task
    def inference(self):
        model_inputs = self.environment.scorer.model.get_inputs(self.environment.data)
        # self.environment.scorer.invoke(input=model_inputs, client=self.client)
        # headers, payload = self.environment.scorer.model.generate_ULCA_payload(
        #     model_inputs, self.environment.scorer.model.config.MODEL_TYPE
        # )
        # response = self.environment.scorer.model.infer(
        #     self.environment.scorer.model.config.HTTP_URL, headers, payload, client=self.client
        # )
        response = self.environment.scorer.model.invoke(input=model_inputs, client=self.client)


if __name__ == "__main__":
    pass

# locust -f locust_new.py --users 1 --spawn-rate 1 -H http://api.dhruva.ai4bharat.org:8090/infer --model=ASRBatchOffConfModel --scorer=MUCSBatchOffConfScorer
