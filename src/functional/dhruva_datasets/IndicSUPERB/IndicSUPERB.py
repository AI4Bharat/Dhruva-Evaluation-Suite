import os
import csv
import json
import base64
import logging
from pathlib import Path
from os import path, listdir
from pydub import AudioSegment
import datasets
import soundfile as sf
import os
import shutil
import tarfile
from pydub import AudioSegment
from .languages import LANGUAGES
from .release_stats import STATS
from tqdm import tqdm
_IndicSUPERB_CITATION = ""
_IndicSUPERB_DESCRIPTION = ""


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



class IndicSUPERBConfig(datasets.BuilderConfig):
    """BuilderConfig for IndicSUPERB."""

    def __init__(self, name, version, language, format, **kwargs):
        """BuilderConfig for IndicSUPERB.
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
        self.format = format

        super(IndicSUPERBConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            **kwargs,
        )
        

class IndicSUPERB(datasets.GeneratorBasedBuilder):
    """The IndicSUPERB benchmark."""

    BUILDER_CONFIGS = [
        IndicSUPERBConfig(
            name=lang,
            version=STATS["version"],
            language=lang,
            description=_IndicSUPERB_DESCRIPTION,
            # features=["audio", "transcript"],
            data_url="",
            citation=_IndicSUPERB_CITATION,
            url="",
            format = "m4a",
        )
        for lang in LANGUAGES
    ]

    def _info(self):
        features = {}
        features["transcript"] = datasets.Value("string")
        features["language"] = datasets.Value("string")
        features["path"] = datasets.Value("string")
        features["audio"] = datasets.features.Audio(sampling_rate=16_000)

        return datasets.DatasetInfo(
            description=_IndicSUPERB_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _IndicSUPERB_CITATION,
            license="",
            version=STATS["version"],
        )
    def _convert_wav(self,path : Path, lang : str):
        # Extract the tar.gz file to a temporary folder
        tar = tarfile.open(path, "r:gz")
        tar.extractall("temp")
        tar.close()
        root = f'temp/{lang}'
        # Convert all M4A files to WAV files
        for file in tqdm(os.listdir(root)):
                if file.endswith(".m4a"):
                    # Load the M4A file using pydub
                    m4a_audio = AudioSegment.from_file(os.path.join(root, file), format="m4a")

                    # Export the M4A audio to a WAV file
                    wav_path = os.path.join(root, os.path.splitext(file)[0] + ".wav")
                    m4a_audio.export(wav_path, format="wav")

                    # Remove the original M4A file
                    os.remove(os.path.join(root, file))


        
        # Open the CSV file for reading and create a CSV reader object
        with open(f'temp/{lang}/transcripts.csv', 'r') as file:
            reader = csv.reader(file)
            
            # Get the index of the "path" header
            header_row = next(reader)
            path_index = header_row.index("path")
            
            # Read the rest of the rows and update the "path" values
            rows = []
            for row in reader:
                row[path_index] = row[path_index].replace('.m4a', '.wav')
                rows.append(row)

        # Open the CSV file for writing and create a CSV writer object
        with open(f'temp/{lang}/transcripts.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the updated header row
            writer.writerow(header_row)
            
            # Write the updated rows
            for row in rows:
                writer.writerow(row)

        # Compress the folder into a new tar.gz file
        with tarfile.open(path, "w:gz") as tar:
            tar.add(f'temp/{lang}', arcname=os.path.basename(f'{lang}'))
        

        # Remove the temporary folder
        shutil.rmtree("temp")

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        base_path = "./dhruva_datasets/IndicSUPERB/data/"
        splits = {"test": {"split": datasets.Split.TEST, "local_extracted_archive": ""}}

        # Download mamager is useful for downloading from ULCA / blob storage
        # archive_path = dl_manager.download(url)
        archive_path = "./data/"
        for folder_name, split in splits.items():
            print(self.config.format)
            if(self.config.format == "m4a"):
                print("converting to wav files")
                self._convert_wav(os.path.join(base_path, folder_name, f"{self.config.language}.tar.gz"), self.config.language )
            
            archive_path = os.path.join(base_path, folder_name, f"{self.config.language}.tar.gz")
            local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None
            split["local_extracted_archive"] = local_extracted_archive

        return [
            datasets.SplitGenerator(
                name=split["split"],
                gen_kwargs={
                    "local_extracted_archive": split["local_extracted_archive"],
                    "archive_iterator": dl_manager.iter_archive(
                        os.path.join(base_path, folder_name, self.config.language + ".tar.gz")
                    ),
                    "metadata_filepath": os.path.join(self.config.language, "transcripts.csv"),
                    "path_to_clips": self.config.language,
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
                # print(path)
                reader = csv.DictReader(lines, delimiter=",", quoting=csv.QUOTE_ALL)
                for row in reader:
                    row["path"] = os.path.join(path_to_clips, row["path"])
                    # if data is incomplete, fill with empty values
                    for field in data_fields:
                        if field not in row:
                            row[field] = ""

                    row["language"] = self.config.language
                    metadata[row["path"]] = row
                print(metadata[row["path"]])
                break
        # print("converting")
        # print("LOCAL EXTRACTER ARCHIVE",local_extracted_archive)
        # for filename in listdir(os.path.join(local_extracted_archive,self.config.language)):
        #     print("converting")
        #     if filename.endswith(".m4a"):
        #         # Load the input file using pydub
        #         sound = AudioSegment.from_file(path.join(path_to_clips, filename), "m4a")

        #         # Create the output filename
        #         output_filename = filename[:-4] + ".wav"
        #         output_file = path.join(path_to_clips, output_filename)
        #         os.remove(path.join(path_to_clips, filename))
        #         # Export the file in WAV format using pydub
        #         sound.export(output_file, format="wav")
        for path, f in archive_iterator:
            # print("PATH: ", path)
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
                    # print("PATH", path)
                    # m4a_file = AudioSegment.from_file(f, "m4a")
                    # wav_file = m4a_file.export(format="wav")
                    result["audio"] = {"path": path, "bytes": f.read()}
                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None
                    yield path, result
