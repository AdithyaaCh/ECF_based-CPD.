import logging

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(filename)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S %z",
    level=logging.INFO,
)

logger = logging.getLogger()
