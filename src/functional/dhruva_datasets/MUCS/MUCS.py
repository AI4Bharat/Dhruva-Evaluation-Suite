import os
import csv
import json
import base64
import logging
from pathlib import Path

import datasets
import soundfile as sf

from .languages import LANGUAGES
from .release_stats import STATS

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
        features["transcript"] = datasets.Value("string")
        features["language"] = datasets.Value("string")
        features["audio"] = (datasets.Audio(sampling_rate=16000),)

        return datasets.DatasetInfo(
            description=_MUCS_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _MUCS_CITATION,
            license="",
            version="0.1.1",
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        base_path = "./dhruva_datasets/MUCS/data/"
        splits = {"test": datasets.Split.TEST}

        # Download mamager is useful for downloading from ULCA / blob storage
        # archive_path = dl_manager.download()
        archive_path = "./data/"
        for folder_name, split in splits.items():
            archive_path = os.path.join(base_path, folder_name, f"{self.config.language}.tar.gz")
            local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "local_extracted_archive": os.path.join(
                        base_path, folder_name, self.config.language
                    ),
                    "archive_iterator": dl_manager.iter_archive(
                        os.path.join(base_path, folder_name, self.config.language + ".tar.gz")
                    ),
                    "metadata_filepath": os.path.join(
                        base_path, folder_name, self.config.language, "transcripts.csv"
                    ),
                    "path_to_clips": os.path.join(
                        base_path, folder_name, self.config.language
                    ),
                },
            )
            for folder_name, split in splits.items()
        ]

    def _generate_examples(
        self,
        local_extracted_archive: str,
        archive_iterator: datasets.DownloadManager.iter_archive,
        metadata_filepath: str,
        path_to_clips: str,
    ):
        """ Generate data """
        data_fields = list(self._info().features.keys())
        metadata = {}
        metadata_found = False
        for path, f in archive_iterator:
            if path == metadata_filepath:
                metadata_found = True
                lines = (line.decode("utf-8") for line in f)
                reader = csv.DictReader(lines, delimiter=",", quoting=csv.QUOTE_NONE)
                for row in reader:
                    row["path"] = os.path.join(path_to_clips, row["path"])
                    # if data is incomplete, fill with empty values
                    for field in data_fields:
                        if field not in row:
                            row[field] = ""
                    metadata[row["path"]] = row
                break

        for path, f in archive_iterator:
            if path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata CSV file."
                if not metadata:
                    break
                if path in metadata:
                    result = dict(metadata[path])
                    # set the audio feature and the path to the extracted file
                    path = (
                        os.path.join(local_extracted_archive, path)
                        if local_extracted_archive
                        else path
                    )
                    result["audio"] = {"path": path, "bytes": f.read()}
                    result["transcript"] = metadata["transcript"]
                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None

                    yield path, result

        # with open(os.path.join(data_dir, transcript_file), encoding="utf-8") as f:
        #     raw_data = csv.reader(f, delimiter=",")
        #     # Skip headers
        #     next(raw_data, None)
        #     for i, row in enumerate(raw_data):
        #         yield i, {
        #                 # "audio": {"bytes": _load_raw(os.path.join(data_dir, row[0])), "path": row[0]},
        #                 "audio": _load_raw(os.path.join(data_dir, row[0])),
        #                 "transcript": row[1],
        #                 "language": self.config.language,
        #             }
