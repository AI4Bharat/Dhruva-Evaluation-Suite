import os
import json
import yaml
import argparse

from datasets import load_dataset

from dhruva_models import DhruvaRESTModel, DhruvaSocketModel
from dhruva_evaluators import DhruvaASREvaluator, DhruvaMTEvaluator, DhruvaTTSEvaluator, DhruvaTransliterationEvaluator
from constants import (
    Enums,
    ULCA_LANGUAGE_CODE_TO_FLORES_MAPPING,
    DATASET_INPUT_COLUMN_MAPPING,
    UserConfiguration,
    EnvSettings,
)
from dhruva_logger import logger


TASK_EVALUATOR_MAPPING = {
    Enums.tasks.NMT: DhruvaMTEvaluator,
    Enums.tasks.ASR: DhruvaASREvaluator,
    Enums.tasks.TTS: DhruvaTTSEvaluator,
    Enums.tasks.Transliteration: DhruvaTransliterationEvaluator
}

MODEL_TYPE_MODEL_MAPPING = {
    Enums.model_type.REST: DhruvaRESTModel,
    Enums.model_type.STREAMING: DhruvaSocketModel
}


def parse_yaml_file(yaml_file_path):
    """Load the YAML file contents into a dictionary"""
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
        if not isinstance(yaml_data, dict):
            raise TypeError(f"Config file not found: {yaml_file_path}")
    return yaml_data


class Evaluation:
    """Run evaluation on a single subset and single model"""

    def __init__(self, user_config):
        self.user_config = user_config
        self.task_evaluator_obj = TASK_EVALUATOR_MAPPING.get(self.user_config.task.type)
        self.model = MODEL_TYPE_MODEL_MAPPING.get(self.user_config.model.type.lower())

    def get_input_column_for_dataset(self, source_language: str, dataset: str):
        dataset_column, lang = None, None
        if dataset == Enums.datasets.FLORES:
            lang = ULCA_LANGUAGE_CODE_TO_FLORES_MAPPING.get(source_language)
            dataset_column = DATASET_INPUT_COLUMN_MAPPING.get(dataset)
            dataset_column = dataset_column + lang
        elif dataset in (Enums.datasets.MUCS, Enums.datasets.IndicSUPERB):
            lang = source_language
            dataset_column = "audio"
            
        else:
            raise KeyError("Can't find input column for given dataset")

        return dataset_column, lang

    def get_label_column_for_dataset(self, target_language: str, dataset: str):
        dataset_column, lang = None, None
        if dataset == Enums.datasets.FLORES:
            lang = ULCA_LANGUAGE_CODE_TO_FLORES_MAPPING.get(target_language)
            dataset_column = DATASET_INPUT_COLUMN_MAPPING.get(dataset)
            dataset_column = dataset_column + lang
        elif dataset in (Enums.datasets.MUCS, Enums.datasets.IndicSUPERB):
            lang = target_language
            dataset_column = "transcript"
        else:
            raise KeyError("Can't find label column for given dataset")
        return dataset_column, lang

    def find_data_subset(self, dataset_source_lang: str, dataset_target_lang: str):
        subset = None
        if self.user_config.task.type == Enums.tasks.NMT:
            subset = dataset_source_lang + "-" + dataset_target_lang
        elif self.user_config.task.type == Enums.tasks.ASR:
            subset = dataset_source_lang
        return subset

    def initialise_dataset_params(
            self,
            source_language: str,
            target_language: str,
            dataset_name: str,
            input_column: str,
            label_column: str,
            subset: str
        ):
        if input_column and label_column and subset:
            self.input_column, self.label_column, self.subset = input_column, label_column, subset
            return

        self.input_column, dataset_source_lang = self.get_input_column_for_dataset(
            source_language, dataset_name
        )
        self.label_column, dataset_target_lang = self.get_label_column_for_dataset(
            target_language, dataset_name
        )
        self.subset = self.find_data_subset(dataset_source_lang, dataset_target_lang)

    def run(self, dataset_path, dataset_name, split):
        # aks = load_dataset("ai4bharat/Aksharantar", self.subset, split=split, streaming=True)
        # print(next(iter(aks)))
        # print(aks)

        self.task_evaluator = self.task_evaluator_obj(
            dataset_name=dataset_name,
            task=self.user_config.task.type,
            default_metric_name=self.user_config.task.metric,
        )
        results = self.task_evaluator.compute(
            model_or_pipeline=self.model(
                url=self.user_config.model.url,
                task=self.user_config.task.type,
                input_column=self.input_column,
                api_key=EnvSettings.api_key,
            ),
            data=dataset_path,  # or dataset path
            # data=aks,
            subset=self.subset,
            split=split,
            input_column=self.input_column,
            label_column=self.label_column,
            metric=self.user_config.task.metric,
        )

        # mucs = load_dataset("./dhruva_datasets/mucs/mucs.py", data_dir="./dhruva_datasets/mucs/data")
        # task_evaluator = DhruvaASREvaluator(task="dhruva_asr", default_metric_name="wer")
        # results = task_evaluator.compute(
        #     model_or_pipeline=DhruvaModel(
        #         url=url,
        #         task="dhruva_asr",
        #         input_column="audio",
        #         input_language_column="language",
        #         api_key="ae66a0b6-69de-4aaf-8fd1-aa07f8ec961b",
        #         format = "wav"
        #     ),
        #     data="./dhruva_datasets/CommonVoice/CommonVoice.py",
        #     # data="ai4bharat/MUCS-internal",
        #     subset="hi",
        #     split="test",
        #     input_column="audio",
        #     label_column="sentence",
        #     metric="wer",
        # )
        logger.warning(f"\n\nResults:\n{json.dumps(results, indent=4)}\n\n\n")
        with open(os.path.join(self.user_config.results_folder, self.subset + ".json"), "w") as f:
            json.dump(results, f)


class EvaluationSuite:
    """Evaluation Suite to run evaluations on multiple subsets of a dataset in one shot"""

    def __init__(self, config_path):
        self.user_config = UserConfiguration.parse_obj(parse_yaml_file(config_path))
        logger.warning(
            f"\n\nUser Config:\n{json.dumps(self.user_config.dict(), indent=4)}\n\n"
        )
        self.evaluator = Evaluation(self.user_config)

    def loop_langs(
            self,
            source_languages,
            target_languages,
            dataset_name,
            dataset_path,
            split,
            input_columns,
            label_columns,
            subsets
        ):
        if input_columns is None or subsets is None or label_columns is None:
            input_columns = [None for i in range(len(source_languages))]
            label_columns = [None for i in range(len(source_languages))]
            subsets = [None for i in range(len(source_languages))]
        print(source_languages, target_languages, input_columns, label_columns, subsets)
        for slang, tlang, inp_col, label_col, subset in zip(
            source_languages, target_languages, input_columns, label_columns, subsets
        ):
            self._run(
                slang,
                tlang,
                dataset_name,
                dataset_path,
                split,
                inp_col,
                label_col,
                subset
            )

    def run_datasets(self):
        for dataset in self.user_config.dataset:
            logger.info(f"Dataset: {dataset}")
            if isinstance(dataset.source_language, list):
                self.loop_langs(
                    dataset.source_language,
                    dataset.target_language,
                    dataset.name,
                    dataset.path,
                    dataset.split,
                    dataset.input_column,
                    dataset.label_column,
                    dataset.subset
                )
            else:
                self._run(
                    dataset.source_language,
                    dataset.target_language,
                    dataset.name,
                    dataset.path,
                    dataset.split,
                    dataset.input_column,
                    dataset.label_column,
                    dataset.subset
                )

    def _run(
            self,
            source_language,
            target_language,
            dataset_name,
            dataset_path, 
            split,
            input_column,
            label_column,
            subset
        ):
        logger.info(
            f"Source Language:{source_language} \
                        \tTarget Language: {target_language}"
        )
        self.evaluator.initialise_dataset_params(
            source_language,
            target_language,
            dataset_name,
            input_column,
            label_column,
            subset
        )
        self.evaluator.run(dataset_path, dataset_name, split)
    
    def run(self):
        if isinstance(self.user_config.dataset, list):
            self.run_datasets()
            # for dataset in self.user_config.dataset:
            #     logger.info(f"Dataset: {dataset}")
            #     if isinstance(dataset.source_language, list):
            #         for slang, tlang in zip(
            #             dataset.source_language, dataset.target_language
            #         ):
            #             logger.info(
            #                 f"Source Language:{slang} \tTarget Language: {tlang}"
            #             )
            #             self.evaluator.initialise_dataset_params(
            #                 slang, tlang, dataset.name, dataset.input_column, dataset.label_column, dataset.subset
            #             )
            #             self.evaluator.run(dataset.path, dataset.name, dataset.split)
            #     else:
            #         logger.info(
            #             f"Source Language:{dataset.source_language} \
            #                      \tTarget Language: {dataset.target_language}"
            #         )
            #         self.evaluator.initialise_dataset_params(
            #             dataset.source_language,
            #             dataset.target_language,
            #             dataset.name,
            #             dataset.input_column,
            #             dataset.label_column,
            #             dataset.subset
            #         )
            #         self.evaluator.run(dataset.path, dataset.name, dataset.split)

        elif isinstance(self.user_config.dataset.source_language, list):
            self.loop_langs(
                self.user_config.dataset.source_language,
                self.user_config.dataset.target_language,
                self.user_config.dataset.name,
                self.user_config.dataset.path,
                self.user_config.dataset.split,
                self.user_config.dataset.input_column,
                self.user_config.dataset.label_column,
                self.user_config.dataset.subset
            )
            # for slang, tlang in zip(
            #     self.user_config.dataset.source_language,
            #     self.user_config.dataset.target_language,
            # ):
            #     logger.info(f"Source Language:{slang} \tTarget Language: {tlang}")
            #     self.evaluator.initialise_dataset_params(
            #         slang,
            #         tlang,
            #         self.user_config.dataset.name,
            #         self.user_config.dataset.input_column,
            #         self.user_config.dataset.label_column,
            #         self.user_config.dataset.subset,
            #     )
            #     self.evaluator.run(
            #         self.user_config.dataset.path,
            #         self.user_config.dataset.name,
            #         self.user_config.dataset.split,
            #     )
        else:
            self._run(
                self.user_config.dataset.source_language,
                self.user_config.dataset.target_language,
                self.user_config.dataset.name,
                self.user_config.dataset.path,
                self.user_config.dataset.split,
                self.user_config.dataset.input_column,
                self.user_config.dataset.label_column,
                self.user_config.dataset.subset,
            )
            # logger.info(
            #     f"Source Language:{self.user_config.dataset.source_language} \
            #                      \tTarget Language: {self.user_config.dataset.target_language}"
            # )
            # self.evaluator.initialise_dataset_params(
            #     self.user_config.dataset.source_language,
            #     self.user_config.dataset.target_language,
            #     self.user_config.dataset.name,
            #     self.user_config.dataset.input_column,
            #     self.user_config.dataset.label_column,
            #     self.user_config.dataset.subset,
            # )
            # self.evaluator.run(
            #     self.user_config.dataset.path,
            #     self.user_config.dataset.name,
            #     self.user_config.dataset.split,
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="")
    args = parser.parse_args()
    suite = EvaluationSuite(args.file)
    suite.run()

