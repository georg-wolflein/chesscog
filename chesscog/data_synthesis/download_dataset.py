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
import sys
import osfclient.cli
import typing
import zipfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from recap import URI
from logging import getLogger

logger = getLogger(__name__)


def _unzip(folder: Path, archive: str):
    logger.info(f"Extracting {archive}")
    with zipfile.ZipFile(folder / archive) as z:
        z.extractall(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the rendered dataset.").parse_args()

    folder = URI("data://render")
    shutil.rmtree(folder, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        logger.info("Downloading rendered dataset from OSF")
        tmp = Path(tmp)
        args = SimpleNamespace(project="xf3ka", output=str(tmp), username=None)
        osfclient.cli.clone(args)
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder.parent, exist_ok=True)
        shutil.move(tmp / "osfstorage", folder)

    # Unzip val and test
    for archive in ("val.zip", "test.zip"):
        _unzip(folder, archive)

    # Unzip train
    logger.info("Merging train dataset")
    command = f"zip -s 0 {folder / 'train.zip'} --out {folder / 'train_full.zip'}"
    if os.waitstatus_to_exitcode(os.system(command)) != 0 or not (folder / "train_full.zip").exists():
        logger.error(
            f"Please manually merge train.zip and train.z01 in {folder}, and unpack the resulting ZIP file to {folder/'train'}.")
        logger.error(
            f"On UNIX systems, the command to merge the ZIP files should look something like \"{command}\".")
        logger.error(
            "Note that the validation and test sets have already been unpacked.")
        sys.exit(-1)
    else:
        for file in ("train.z01", "train.zip"):
            (folder / file).unlink()
        shutil.move(folder / "train_full.zip", folder / "train.zip")
        _unzip(folder, "train.zip")
