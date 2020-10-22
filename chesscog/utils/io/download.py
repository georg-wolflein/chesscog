from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import shutil
from pathlib import Path
import logging
import typing
import tempfile
import os

from .path_manager import URI

logger = logging.getLogger(__name__)


def _get_members(f: zipfile.ZipFile) -> typing.Iterator[zipfile.ZipInfo]:
    parts = [name.partition("/")[0]
             for name in f.namelist()
             if not name.endswith("/")]

    prefix = os.path.commonprefix(parts)
    if prefix:
        prefix += "/"
        offset = len(prefix)
    # Alter file names
    for zipinfo in f.infolist():
        name = zipinfo.filename
        if len(name) > offset:
            zipinfo.filename = name[offset:]
            yield zipinfo


def download_zip_folder_from_google_drive(file_id: str, destination: os.PathLike, show_size: bool = False, skip_if_exists: bool = True):
    destination = URI(destination)
    if skip_if_exists and destination.exists():
        logger.info(
            f"Not downloading {file_id} to {destination} again because it already exists")
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file = Path(tmp_dir) / f"{destination.name}.zip"
        logger.info(f"Downloading {file_id} to {zip_file}")
        gdd.download_file_from_google_drive(file_id="1XClmGJwEWNcIkwaH0VLuvvAY3qk_CRJh",
                                            dest_path=zip_file,
                                            overwrite=True,
                                            showsize=show_size)
        logger.info(f"Unzipping {zip_file} to {destination}")
        shutil.rmtree(destination, ignore_errors=True)
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(destination, _get_members(f))
        logger.info(f"Finished downloading {file_id} to {destination}")
