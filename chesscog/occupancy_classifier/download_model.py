"""Script to download the best occupancy classifier (already trained).

Running this script will download the classifier that was used in the report.
It will be downloaded to ``models://occupancy_classifier``.

.. code-block:: console

    $ python -m chesscog.occupancy_classifier.download_model --help
    usage: download_model.py [-h]
    
    Download the occupancy classifier.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import functools
from chesscog.core.io import download_zip_folder
import argparse


ensure_model = functools.partial(download_zip_folder,
                                 "https://github.com/georg-wolflein/chesscog/releases/download/0.1.0/occupancy_classifier.zip",
                                 "models://occupancy_classifier")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Download the occupancy classifier.").parse_args()
    ensure_model(show_size=True)
