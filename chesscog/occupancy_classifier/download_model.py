"""Script to download the best occupancy classifier."""

from chesscog.utils.io import URI, download_zip_folder_from_google_drive

download_zip_folder_from_google_drive("1mLaSIh7KzGeMuyGmhtwIYVVVN6l_eC1k",
                                      "models://occupancy_classifier",
                                      show_size=True)
