"""Script to download the best occupancy classifier."""

import functools
from chesscog.utils.io import download_zip_folder_from_google_drive


ensure_model = functools.partial(download_zip_folder_from_google_drive,
                                 "1MUGB_OlcbHFyHh2BwtkXXi6ZBPeEfHtQ",
                                 "models://occupancy_classifier")


if __name__ == "__main__":
    ensure_model(show_size=True)
