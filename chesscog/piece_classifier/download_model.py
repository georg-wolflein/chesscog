"""Script to download the best piece classifier (already trained).

Running this script will download the classifier that was used in the report.
It will be downloaded to ``models://piece_classifier``.

.. code-block:: console

    $ python -m chesscog.piece_classifier.download_model --help
    usage: download_model.py [-h]
    
    Download the piece classifier.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import functools
from chesscog.core.io import download_zip_folder
import argparse

ensure_model = functools.partial(download_zip_folder,
                                 "https://github.com/georg-wolflein/chesscog/releases/download/0.1.0/piece_classifier.zip",
                                 "models://piece_classifier")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Download the piece classifier.").parse_args()
    ensure_model(show_size=True)
