import sys
import logging
from logging import getLogger


LOGGER = getLogger("ml_summarization")

for h in LOGGER.handlers:
    LOGGER.removeHandler(h)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                              '%m-%d-%Y %H:%M:%S')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

LOGGER.addHandler(stdout_handler)

