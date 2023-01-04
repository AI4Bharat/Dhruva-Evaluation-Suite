"""
Decorators for:
    * Timing
    * Logging
    * Input formatting
    * Output formatting

"""

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
