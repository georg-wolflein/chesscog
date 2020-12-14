"""Script to download the best piece classifier."""

import functools
from chesscog.core.io import download_zip_folder


ensure_models = functools.partial(download_zip_folder,
                                  "https://github.com/georgw777/chesscog/releases/download/0.1.0/transfer_learning_models.zip",
                                  "models://transfer_learning")


if __name__ == "__main__":
    ensure_models(show_size=True)
