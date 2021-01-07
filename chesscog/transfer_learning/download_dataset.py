"""Script to download the sample dataset of a different chess set.

.. code-block:: console

    $ python -m chesscog.transfer_learning.download_dataset --help
    usage: download_dataset.py [-h]
    
    Download the sample transfer learning dataset.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import functools
import argparse

from chesscog.core.io import download_zip_folder_from_google_drive

ensure_dataset = functools.partial(download_zip_folder_from_google_drive,
                                   "1Z9fTXRb7FlqzgTTXoywgiQP-Z1cH1v3W",
                                   "data://transfer_learning/images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the sample transfer learning dataset.").parse_args()
    ensure_dataset(show_size=True)
