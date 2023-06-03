"""Script to prepare the inference results as a confusion matrix for use in a LaTeX table.

.. code-block:: console

    $ python -m chesscog.report.prepare_confusion_matrix --help  
    usage: prepare_confusion_matrix.py [-h] [--results RESULTS]
                                       [--dataset DATASET]
    
    Prepare confusion matrix for LaTeX
    
    optional arguments:
      -h, --help         show this help message and exit
      --results RESULTS  parent results folder
      --dataset DATASET  the dataset to evaluate
"""

import chess
import numpy as np
import typing
import pandas as pd
import argparse
from recap import URI
import sys

CATEGORIES = [
    "P", "N", "B", "R", "Q", "K",
    "p", "n", "b", "r", "q", "k",
    None
]

LATEX_HEADINGS = [
    "\\WhitePawnOnWhite",
    "\\WhiteKnightOnWhite",
    "\\WhiteBishopOnWhite",
    "\\WhiteRookOnWhite",
    "\\WhiteQueenOnWhite",
    "\\WhiteKingOnWhite",
    "\\BlackPawnOnWhite",
    "\\BlackKnightOnWhite",
    "\\BlackBishopOnWhite",
    "\\BlackRookOnWhite",
    "\\BlackQueenOnWhite",
    "\\BlackKingOnWhite",
    "\\phantom{\\WhitePawnOnWhite}"
]
LATEX_HEADINGS = [("\\raisebox{-.2cm}{" + x + "}")
                  for x in LATEX_HEADINGS]


def _get_category(piece: typing.Union[chess.Piece, None]) -> str:
    if piece is None:
        return None
    return piece.symbol()


def _get_confusion_matrix(predicted: chess.Board, actual: chess.Board) -> np.ndarray:
    matrix = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=np.int32)
    for square in chess.SQUARES:
        pred = _get_category(predicted.piece_at(square))
        act = _get_category(actual.piece_at(square))
        matrix[CATEGORIES.index(pred), CATEGORIES.index(act)] += 1
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare confusion matrix for LaTeX")
    parser.add_argument("--results", help="parent results folder",
                        type=str, default="results://recognition")
    parser.add_argument("--dataset", help="the dataset to evaluate",
                        type=str, default="train")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(URI(args.results) / f"{args.dataset}.csv")
    if "fen_predicted" not in df.columns or "fen_actual" not in df.columns:
        print(
            "FEN columns not found in the CSV; ensure you export the CSV using --save-fens")
        sys.exit(-1)

    # Filter out samples where the corners could not be detected
    df = df[(df["num_incorrect_corners"] != 4) | (df["error"] != "None")]

    matrix = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=np.int32)
    for i, row in df.iterrows():
        actual = chess.Board(row.fen_actual)
        predicted = chess.Board(row.fen_predicted)
        matrix += _get_confusion_matrix(predicted, actual)

    print("& " + " \n& ".join(LATEX_HEADINGS) + " \\\\\n")

    def convert_cell(value: int) -> str:
        result = f"{value:5d}"
        color = "\\cellcolor{black!20} "
        if value == 0:
            color = " " * len(color)
        return color + result
    for i, row in enumerate(matrix):
        print(f"{LATEX_HEADINGS[i]:50s} & " +
              " & ".join(map(convert_cell, row)) + " \\\\")
