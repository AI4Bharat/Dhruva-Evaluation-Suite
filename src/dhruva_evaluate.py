import logging
from evaluate import evaluator
from datasets import load_dataset, Dataset
from dhruva_datasets import MUCS
from dhruva_models import DhruvaModel
from dhruva_evaluators import DhruvaASREvaluator


def dhruva_evaluate(*args, **kwargs):
    url = "https://api.dhruva.ai4bharat.org/services/inference/asr?serviceId=ai4bharat/conformer-hi-gpu--t4"

    # mucs = load_dataset("./dhruva_datasets/MUCS", split="test")
    task_evaluator = DhruvaASREvaluator()
    results = task_evaluator.compute(
        model_or_pipeline=DhruvaModel(
            url=url,
            task="dhruva-asr",
            input_column="audio",
            language_column="language",
            api_key="ae66a0b6-69de-4aaf-8fd1-aa07f8ec961b"
        ),
        data="./dhruva_datasets/MUCS/MUCS.py",
        # data=mucs,
        subset="MUCS-hi",
        split="test",
        input_column="audio",
        label_column="transcript",
        metric="wer",
    )
    print("----> ", results)


if __name__ == "__main__":
    # audio_path = "/Users/ashwin/ai4b/perf_testing/Dhruva-Evaluation-Suite/datasets/raw/MUCS/Hindi"
    # dataset = load_dataset("./dhruva_datasets/asr.py", "MUCS-hi", data_dir=audio_path, split="test")  # , streaming=True
    # print(dataset.column_names, dataset.shape, isinstance(dataset, Dataset))
    # print(dataset["audio"])
    # dataset = dataset[:10]

    
    # task_evaluator = evaluator("automatic-speech-recognition")
    dhruva_evaluate()


# Load using custom script
# Structure:
# {
#     "audio":{"path": "", "array": [], "sampling_rate": 16000},
#     "client_id": 1,
#     "locale": "hi",
#     "age": 40,
#     "accent": "",
#     "gender": "",
#     "segment": "",
#     "transcript": ""
# }

# Hit Dhruva
# Gen WER

# NMT
# Structure

# Hit Dhruva
# Gen BLEU