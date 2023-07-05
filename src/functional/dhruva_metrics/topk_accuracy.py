import evaluate
import datasets

_KWARGS_DESCRIPTION = """
"""

_CITATION = """
"""

_DESCRIPTION = """
"""


class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Value("string"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"
            ],
        )

    def _compute(self, predictions, references):
        total = 0
        correct = 0
        # print(predictions)
        for prediction, reference in zip(predictions, references):
            total += 1
            if reference in prediction:
                correct += 1
        return correct / total
