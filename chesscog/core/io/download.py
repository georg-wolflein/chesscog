"""Module for downloading and extracting ZIP files from various sources.
"""

from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import shutil
from pathlib import Path
import logging
import typing
import tempfile
import os
import requests
from tqdm import tqdm
from recap import URI

logger = logging.getLogger(__name__)


def _get_members(f: zipfile.ZipFile) -> typing.Iterator[zipfile.ZipInfo]:
    parts = [name.partition("/")[0]
             for name in f.namelist()
             if not name.endswith("/")
             and not ".DS_Store" in name
             and not "__MACOSX" in name]

    prefix = os.path.commonprefix(parts)
    if "/" in prefix:
        prefix = prefix[:prefix.rfind("/") + 1]
    else:
        prefix = ""
    offset = len(prefix)
    # Alter file names
    for zipinfo in f.infolist():
        name = zipinfo.filename
        if ".DS_Store" in name or "__MACOSX" in name:
            continue
        if len(name) > offset:
            zipinfo.filename = name[offset:]
            yield zipinfo


def download_file(url: str, destination: os.PathLike, show_size: bool = False):
    """Download a file from a URL to a destination.

    Args:
        url (str): the URL
        destination (os.PathLike): the destination
        show_size (bool, optional): whether to display a progress bar. Defaults to False.
    """
    destination = URI(destination)
    logger.info(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    if show_size:
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
    with destination.open("wb") as f:
        for data in response.iter_content(block_size):
            if show_size:
                progress_bar.update(len(data))
            f.write(data)
    if show_size:
        progress_bar.close()


def download_zip_folder(url: str, destination: os.PathLike, show_size: bool = False, skip_if_exists: bool = True):
    """Download and extract a ZIP folder from a URL.

    The file is first downloaded to a temporary location and then extracted to the target folder.

    Args:
        url (str): the URL of the ZIP file
        destination (os.PathLike): the destination folder
        show_size (bool, optional): whether to display a progress bar. Defaults to False.
        skip_if_exists (bool, optional): if true, will do nothing when the destination path exists already. Defaults to True.
    """
    destination = URI(destination)
    if skip_if_exists and destination.exists():
        logger.info(
            f"Not downloading {url} to {destination} again because it already exists")
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file = Path(tmp_dir) / f"{destination.name}.zip"
        download_file(url, zip_file, show_size)
        logger.info(f"Unzipping {zip_file} to {destination}")
        shutil.rmtree(destination, ignore_errors=True)
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(destination, _get_members(f))
        logger.info(f"Finished downloading {url} to {destination}")


def download_zip_folder_from_google_drive(file_id: str, destination: os.PathLike, show_size: bool = False, skip_if_exists: bool = True):
    """Download and extract a ZIP file from Google Drive.

    Args:
        file_id (str): the Google Drive file ID
        destination (os.PathLike): the destination folder
        show_size (bool, optional): whether to display a progress bar. Defaults to False.
        skip_if_exists (bool, optional): if true, will do nothing when the destination path exists already. Defaults to True.
    """
    destination = URI(destination)
    if skip_if_exists and destination.exists():
        logger.info(
            f"Not downloading {file_id} to {destination} again because it already exists")
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file = Path(tmp_dir) / f"{destination.name}.zip"
        logger.info(f"Downloading {file_id} to {zip_file}")
        gdd.download_file_from_google_drive(file_id=file_id,
                                            dest_path=zip_file,
                                            overwrite=True,
                                            showsize=show_size)
        logger.info(f"Unzipping {zip_file} to {destination}")
        shutil.rmtree(destination, ignore_errors=True)
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(destination, _get_members(f))
        logger.info(f"Finished downloading {file_id} to {destination}")
