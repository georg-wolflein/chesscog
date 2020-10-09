from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR", Path.home() / "chess_data"))
