from locust import HttpUser, between, task

import EvaluatorBase

class LocustEvaluator(EvaluatorBase):

    def eval(self) -> None:
        prev_output = None
        for name, plugin in self._plugins.items():
            logging.info(name, plugin)

            plugin(**self.config)

            if "model" in name.lower():
                self.config["model"] = plugin
                continue
            elif "postprocessor" in name.lower():
                self.config["postprocessor"] = plugin
                continue
            elif "metrics" in name.lower():
                self.config["metrics"] = plugin
                continue

            stage_output = plugin.invoke()
            if "dataset" in name:
                self.config["dataset_output"] = stage_output
            elif "preprocessor" in name:
                self.config["preprocessed_output"] = stage_output
            elif "scorer" in name:
                self.config["scorer_output"] = stage_output
            else:
                pass


if __name__ == "__main__":
    eval - Evaluator(
        {
            "plugins": [
                "ASRPreprocessor"
                "ASRBatchE2EModel",
                "ASRBatchE2EScorer"
            ]
        }
    )
    eval.eval()