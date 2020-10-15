import sys
import logging


def _setup_logger(level: int = logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


_setup_logger()
