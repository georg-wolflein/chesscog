"""Script to download the rendered dataset.

.. code-block:: console

    $ python -m chesscog.data_synthesis.download_dataset --help
    usage: download_dataset.py [-h]
    
    Download the rendered dataset.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import functools
import argparse
import shutil
import os
import osfclient.cli
import typing
import zipfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from recap import URI
from logging import getLogger

logger = getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the rendered dataset.").parse_args()

    folder = URI("data://render")
    with tempfile.TemporaryDirectory() as tmp:
        logger.info("Downloading rendered dataset from OSF")
        tmp = Path(tmp)
        args = SimpleNamespace(project="xf3ka", output=str(tmp), username=None)
        osfclient.cli.clone(args)
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder.parent, exist_ok=True)
        shutil.move(tmp / "osfstorage", folder)
    logger.info("Merging train dataset")
    try:
        os.system(
            f"zip -s 0 {folder / 'train.zip'} --out {folder / 'train_full.zip'}")
    except Exception:
        raise Exception(f"Please manually unpack the ZIP archives at {folder}")
    for file in ("train.z01", "train.zip"):
        (folder / file).unlink()
    shutil.move(folder / "train_full.zip", folder / "train.zip")
    for archive in ("train.zip", "val.zip", "test.zip"):
        logger.info(f"Extracting {archive}")
        with zipfile.ZipFile(folder / archive) as z:
            z.extractall(folder)
