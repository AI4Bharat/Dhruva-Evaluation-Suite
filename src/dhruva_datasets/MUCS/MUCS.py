import os
import csv
import json
import base64
import logging
from pathlib import Path

import datasets
import soundfile as sf


_MUCS_CITATION = ""
_MUCS_DESCRIPTION = ""


def _load_wav(path: Path):
    audio, _ = sf.read(path)
    return audio.tolist()

def _load_raw(path: str):
    with open(path, "rb") as ip:
        # HF datasets don't have a bytes feature
        # Sending bytes as string corrupts base64 conversion later
        # Decoding utf-8 is throwing errors
        # Sending it as a base64 string
        # return base64.b64encode(ip.read()).decode("utf-8")
        return ip.read()  # .decode("utf-8")


class MUCSConfig(datasets.BuilderConfig):
    """BuilderConfig for MUCS."""

    def __init__(self, features, data_url, citation, url, language, **kwargs):
        """BuilderConfig for MUCS.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          **kwargs: keyword arguments forwarded to super.
        """
        super(MUCSConfig, self).__init__(version=datasets.Version("1.0.1"), **kwargs)
        self.features = features
        self.data_url = data_url
        self.citation = citation
        self.url = url
        self.language = language


class MUCS(datasets.GeneratorBasedBuilder):
    """The MUCS benchmark."""

    BUILDER_CONFIGS = [
        MUCSConfig(
            name="MUCS-hi",
            language="hi",
            description=_MUCS_DESCRIPTION,
            features=["audio", "transcript"],
            data_url="",
            citation=_MUCS_CITATION,
            url="",
        ),
    ]

    def _info(self):
        # features = {feature: datasets.Value("string") for feature in self.config.features}
        features = {}
        if self.config.language == "hi":
            # features["audio"] = datasets.Value("string")
            features["transcript"] = datasets.Value("string")
            features["language"] = datasets.Value("string")
            features["audio"] = datasets.Audio(sampling_rate=16000),

        return datasets.DatasetInfo(
            description=_MUCS_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _MUCS_CITATION,
            license="",
            version="0.1.1"
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        # Download mamager is iseful for downloading from ULCA / blob storage
        # dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        base_path = "../Dhruva-Evaluation-Suite/datasets/raw/MUCS/"
        dl_dir = os.path.join(base_path, self.config.language)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "transcript_file": "transcripts.csv",
                    "data_dir": dl_dir,
                    "dl_manager": dl_manager,
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(
        self, data_dir: str, transcript_file: str, split: str, dl_manager: datasets.DownloadManager
    ):
        if split != datasets.Split.TEST:
            logging.warn(f"{self.config.name} contains only Test split")

        with open(os.path.join(data_dir, transcript_file), encoding="utf-8") as f:
            raw_data = csv.reader(f, delimiter=",")
            # Skip headers
            next(raw_data, None)
            for i, row in enumerate(raw_data):
                yield i, {
                        "audio": {"bytes": _load_raw(os.path.join(data_dir, row[0])), "path": row[0]},
                        # "audio": _load_raw(os.path.join(data_dir, row[0])),
                        "transcript": row[1],
                        "language": self.config.language,
                    }
