"""Core file system operations.

This module is included by default to set up the paths for use with :class:`recap.URI`.
"""

from pathlib import Path
import os
from recap.path_manager import register_translator

from .download import download_file, download_zip_folder, download_zip_folder_from_google_drive


_DATA_DIR = Path(os.getenv("DATA_DIR",
                           Path.home() / "chess_data"))
_CONFIG_DIR = Path(os.getenv("CONFIG_DIR",
                             Path(__file__).parent.parent.parent.parent / "config"))
_RUNS_DIR = Path(os.getenv("RUNS_DIR",
                           Path(__file__).parent.parent.parent.parent / "runs"))
_RESULTS_DIR = Path(os.getenv("RESULTS_DIR",
                              Path(__file__).parent.parent.parent.parent / "results"))
_MODELS_DIR = Path(os.getenv("MODELS_DIR",
                             Path(__file__).parent.parent.parent.parent / "models"))
_REPORT_DIR = Path(os.getenv("REPORT_DIR",
                             Path(__file__).parent.parent.parent.parent.parent / "chess-recognition-report"))


register_translator("data", _DATA_DIR)
register_translator("config", _CONFIG_DIR)
register_translator("runs", _RUNS_DIR)
register_translator("results", _RESULTS_DIR)
register_translator("models", _MODELS_DIR)
register_translator("report", _REPORT_DIR)
