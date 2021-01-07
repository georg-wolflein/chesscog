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

from chesscog.core.io import download_zip_folder_from_google_drive

ensure_dataset = functools.partial(download_zip_folder_from_google_drive,
                                   "1XClmGJwEWNcIkwaH0VLuvvAY3qk_CRJh",
                                   "data://render")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the rendered dataset.").parse_args()
    ensure_dataset(show_size=True)
