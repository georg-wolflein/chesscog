"""Script to extract FEN positions from the ``data://games.pgn`` database of chess game.

.. code-block:: console

    $ python -m chesscog.data_synthesis.create_fens --help     
    usage: create_fens.py [-h]
    
    Create the fens.txt file by selecting 2%% of the positions from
    games.pgn.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import chess.pgn
from pathlib import Path
import numpy as np
import argparse

from recap import URI


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Create the fens.txt file by selecting 2%% of the positions from games.pgn.").parse_args()
    dataset_path = URI("data://games.pgn")
    fens_path = URI("data://fens.txt")

    fens = set()
    with dataset_path.open("r") as pgn:
        while (game := chess.pgn.read_game(pgn)) is not None:
            board = game.board()
            moves = list(game.mainline_moves())
            moves_mask = np.random.randint(0, 50, len(moves)) == 0
            for move, mask in zip(moves, moves_mask):
                board.push(move)
                if mask:
                    color = "W" if board.turn == chess.WHITE else "B"
                    fens.add(color + board.board_fen())

    with fens_path.open("w") as f:
        for fen in fens:
            f.write(fen + "\n")
