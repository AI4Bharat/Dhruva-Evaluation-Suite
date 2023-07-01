import os
import csv
import json
import pathlib
import logging

import datasets

from .languages import LANGUAGES
from .release_stats import STATS

_MUCS_CITATION = ""
_MUCS_DESCRIPTION = ""
_DATA_URL = "https://huggingface.co/datasets/ai4bharat/mucs-internal/resolve/main/data"


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
            data_url="",
            citation=_MUCS_CITATION,
            url="",
        )
        for lang in LANGUAGES
    ]

    def _info(self):
        features = {}
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
        splits = {"test": datasets.Split.TEST}
        audio_path = {}
        metadata_path = {}
        local_extracted_archive = {}

        for _, split in splits.items():
            audio_paths = f"{_DATA_URL}/{self.config.language}/audio_{split}.tar.gz"
            audio_path[split] = dl_manager.download(audio_paths)
            local_extracted_archive[split] = dl_manager.extract(audio_path[split]) if not dl_manager.is_streaming else None
            metadata_path[split] = dl_manager.download_and_extract(f"{_DATA_URL}/{self.config.language}/transcripts_{split}.csv")

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive[split],
                    "archive_iterator": dl_manager.iter_archive(audio_path[split]),
                    "metadata_filepath": metadata_path[split],
                    "path_to_clips": self.config.language,
                },
            )
            for _, split in splits.items()
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

        with open(metadata_filepath, "r") as metadata_f:
            reader = csv.DictReader(metadata_f)
            for item in reader:
            # for item in meta:
                meta_item = {}
                for field in data_fields:
                    if field not in item:
                        meta_item[field] = ""

                # meta_item["path"] = os.path.join(path_to_clips, item["audioFilename"])
                meta_item["path"] = os.path.join(path_to_clips, item["path"])
                meta_item["transcript"] = item["transcript"]
                meta_item["language"] = self.config.language
                metadata[meta_item["path"]] = meta_item

        for path, f in archive_iterator:
            rel_path = os.path.join(*pathlib.Path(path).parts[-2:])
            if rel_path.startswith(path_to_clips):
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
