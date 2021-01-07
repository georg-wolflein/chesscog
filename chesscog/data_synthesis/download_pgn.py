"""Script to download Magnus Carlsen's chess games to ``data://games.pgn``.

.. code-block:: console

    $ python -m chesscog.data_synthesis.download_pgn --help
    usage: download_pgn.py [-h]
    
    Download Magnus Carlsen's chess games to data://games.pgn.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import urllib.request
import zipfile
import argparse

from recap import URI

if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Download Magnus Carlsen's chess games to data://games.pgn.").parse_args()
    zip_file = URI("data://games.zip")
    urllib.request.urlretrieve("https://www.pgnmentor.com/players/Carlsen.zip",
                               zip_file)
    with zipfile.ZipFile(zip_file) as zip_f:
        with zip_f.open("Carlsen.pgn", "r") as in_f, URI("data://games.pgn").open("wb") as out_f:
            out_f.write(in_f.read())
