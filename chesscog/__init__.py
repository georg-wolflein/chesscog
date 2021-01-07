"""Chess position inference using computer vision.
"""

import sys
import logging

from .core import io as _
from .__version__ import __version__


def _setup_logger(level: int = logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


_setup_logger()
