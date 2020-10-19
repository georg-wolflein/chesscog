import chess.pgn
from pathlib import Path
import numpy as np

from chesscog.utils.io import URI

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
