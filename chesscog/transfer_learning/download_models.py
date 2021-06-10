"""Script to download the fine-tuned classifiers on the sample dataset.

Running this script will download both classifiers used in the report in the chapter "Adapting to new chess sets".
They will be downloaded to ``models://transfer_learning``.

.. code-block:: console

    $ python -m chesscog.transfer_learning.download_models --help 
    usage: download_models.py [-h]
    
    Download the fine-tuned piece and occupancy classifiers.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import functools
import argparse

from chesscog.core.io import download_zip_folder


ensure_models = functools.partial(download_zip_folder,
                                  "https://github.com/georg-wolflein/chesscog/releases/download/0.2.7/transfer_learning_models.zip",
                                  "models://transfer_learning")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Download the fine-tuned piece and occupancy classifiers.").parse_args()
    ensure_models(show_size=True)
