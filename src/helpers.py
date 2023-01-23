"""
Decorators for:
    * Timing
    * Logging
    * Input formatting
    * Output formatting

"""


import shutil
import logging
import functools
from config import BaseConfig


# create and configure main logger
logger = logging.getLogger(BaseConfig.LOGGER)
logger.setLevel(BaseConfig.LOGGING_LEVEL)
# create console handler with a higher log level
handler = logging.StreamHandler()
handler.setLevel(BaseConfig.LOGGING_LEVEL)
# create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(handler)


def class_factory(class_name, bases, config):
    if not isinstance(bases, tuple):
        bases = tuple(bases)
    return type(class_name, bases, config)


def extract_files(input_file, extract_path):
    if ".tar" in input_file.suffixes:  # Expecting Path objects. Not filenames / strs
        untar()
    if ".zip" in input_file.suffixes:  # Expecting Path objects. Not filenames / strs
        shutil.unpack_archive(input_file, extract_path)

def untar(input_file, extract_path):
    if ".gz" in input_file.suffixes:
        tar = tarfile.open(input_file, "r:gz")
        tar.extractall(path=extract_path)
        tar.close()
    elif ".tar" in input_file.suffixes:
        tar = tarfile.open(input_file, "r:")
        tar.extractall(path=extract_path)
        tar.close()
    else:
        raise TypeError(f"Check File type for {inp_file}. Did you mean .tar or .tar.gz?")
