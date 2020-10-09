from pathlib import Path
import os

if "DATA_DIR" in os.environ:
    DATA_DIR = os.environ["DATA_DIR"]
else:
    DATA_DIR = Path(__file__).parent.parent / "data"
