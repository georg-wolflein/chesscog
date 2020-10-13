import sys
import logging
from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR",
                          Path.home() / "chess_data"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR",
                            Path(__file__).parent.parent / "config"))
RUNS_DIR = Path(os.getenv("RUNS_DIR",
                          Path(__file__).parent.parent / "runs"))


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
