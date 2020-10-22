"""Script to download the best occupancy classifier."""

from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import shutil
import logging

from chesscog.utils.io import URI

logger = logging.getLogger(__name__)


def ensure_model(showsize: bool = False):
    """Download the model if it is not already present.
    """
    destination = URI("models://occupancy_classifier.pt")
    if destination.exists():
        logger.debug(f"Model already exists at {destination}")
    else:
        logger.info(f"Downloading model to {destination}...")
        gdd.download_file_from_google_drive(file_id="1mLaSIh7KzGeMuyGmhtwIYVVVN6l_eC1k",
                                            dest_path=destination,
                                            overwrite=True,
                                            showsize=showsize)


if __name__ == "__main__":
    ensure_model(showsize=True)
