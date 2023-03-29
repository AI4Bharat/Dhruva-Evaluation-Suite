import os
import csv
import json
import base64
import pathlib
import logging

import datasets
import soundfile as sf

from .languages import LANGUAGES
from .release_stats import STATS

_MUCS_CITATION = ""
_MUCS_DESCRIPTION = ""
_DATA_URL = "https://huggingface.co/datasets/ai4bharat/mucs-internal/resolve/main/data"


def _load_wav(path: pathlib.Path):
    audio, _ = sf.read(path)
    return audio.tolist()


def _load_raw(path: str):
    with open(path, "rb") as ip:
        # HF datasets don't have a bytes feature
        # Sending bytes as string corrupts base64 conversion later
        # Decoding utf-8 is throwing errors
        # Sending it as a base64 string
        return base64.b64encode(ip.read()).decode("utf-8")
        # return ip.read()  # .decode("utf-8")


class MUCSConfig(datasets.BuilderConfig):
    """BuilderConfig for MUCS."""

    def __init__(self, name, version, language, **kwargs):
        """BuilderConfig for MUCS.
        Args:
            name: Dataset name.
            version: Dataset name.
            language: Current Language chosen
            **kwargs: keyword arguments forwarded to super.
        """

        self.name = name
        self.language = language
        self.data_url = kwargs.pop("data_url")
        self.citation = kwargs.pop("citation")
        self.url = kwargs.pop("url")

        super(MUCSConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            **kwargs,
        )


class MUCS(datasets.GeneratorBasedBuilder):
    """The MUCS benchmark."""

    BUILDER_CONFIGS = [
        MUCSConfig(
            name=lang,
            version=STATS["version"],
            language=lang,
            description=_MUCS_DESCRIPTION,
            # features=["audio", "transcript"],
            data_url="",
            citation=_MUCS_CITATION,
            url="",
        )
        for lang in LANGUAGES
    ]

    def _info(self):
        # features = {feature: datasets.Value("string") for feature in self.config.features}
        features = {}
        # if self.config.language == "hi":
        # features["audio"] = datasets.Value("string")
        features["path"] = datasets.Value("string")
        features["transcript"] = datasets.Value("string")
        features["language"] = datasets.Value("string")
        features["audio"] = datasets.Audio(sampling_rate=16000)

        return datasets.DatasetInfo(
            description=_MUCS_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _MUCS_CITATION,
            license="",
            version="0.1.1",
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        base_path = "data/"
        splits = {"test": datasets.Split.TEST}
        audio_path = {}
        metadata_path = {}

        for folder_name, split in splits.items():
            audio_paths = f"{_DATA_URL}/{self.config.language}/audio_{split}.tar.gz"
            audio_path[split] = dl_manager.download(audio_paths)
            local_extracted_archive = dl_manager.extract(audio_path[split]) if not dl_manager.is_streaming else None
            metadata_path[split] = f"{_DATA_URL}/{self.config.language}/data_{split}.json"

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(audio_path[split]),
                    "metadata_filepath": metadata_path[split],
                    "path_to_clips": self.config.language,
                },
            )
            for folder_name, split in splits.items()
        ]

    def _generate_examples(
        self,
        local_extracted_archive,
        archive_iterator: datasets.DownloadManager.iter_archive,
        metadata_filepath: str,
        path_to_clips: str,
    ):
        """ Generate data """
        data_fields = list(self._info().features.keys())
        metadata = {}
        metadata_found = False
        with open(metadata_filepath, "r") as metadata_f:
            meta = json.load(metadata_f)
            for item in meta:
                meta_item = {}
                for field in data_fields:
                    if field not in item:
                        meta_item[field] = ""

                meta_item["path"] = os.path.join(path_to_clips, item["audioFilename"])
                metadata[meta_item["path"]] = meta_item
                meta_item["transcript"] = item["text"]
                meta_item["language"] = self.config.language
            metadata_found = True

        for path, f in archive_iterator:
            rel_path = os.path.join(*pathlib.Path(path).parts[-2:])
            if rel_path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata CSV file."
                if not metadata:
                    break
                if path in metadata:
                    result = dict(metadata[rel_path])
                    # set the audio feature and the path to the extracted file
                    path = (
                        os.path.join(local_extracted_archive, rel_path)
                        if local_extracted_archive
                        else path
                    )

                    result["audio"] = {"path": path, "bytes": f.read()}
                    yield path, result
