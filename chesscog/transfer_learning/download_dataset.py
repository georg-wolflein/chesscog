"""Script to download the rendered dataset."""

import functools
from chesscog.core.io import download_zip_folder_from_google_drive

ensure_dataset = functools.partial(download_zip_folder_from_google_drive,
                                   "1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W",
                                   "data://transfer_learning/images")

if __name__ == "__main__":
    ensure_dataset(show_size=True, skip_if_exists=False)
