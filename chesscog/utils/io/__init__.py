from pathlib import Path
import os

from .path_manager import PathManager, URI, PathTranslator
from .download import download_zip_folder_from_google_drive


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


def _register_translator(scheme: str, path: Path):
    class Translator(PathTranslator):
        def __init__(self):
            super().__init__(path)
    PathManager.register_handler(scheme)(Translator())


_register_translator("data", _DATA_DIR)
_register_translator("config", _CONFIG_DIR)
_register_translator("runs", _RUNS_DIR)
_register_translator("results", _RESULTS_DIR)
_register_translator("models", _MODELS_DIR)
_register_translator("report", _REPORT_DIR)
