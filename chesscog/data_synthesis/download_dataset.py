"""Script to download the rendered dataset."""

import functools
from chesscog.core.io import download_zip_folder_from_google_drive

ensure_dataset = functools.partial(download_zip_folder_from_google_drive,
                                   "1XClmGJwEWNcIkwaH0VLuvvAY3qk_CRJh",
                                   "data://render")

if __name__ == "__main__":
    ensure_dataset(show_size=True)
