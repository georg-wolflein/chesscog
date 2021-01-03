"""Script to output some information about the distribution of errors in the recognition.

.. code-block:: console

    $ python -m chesscog.report.prepare_error_distribution --help
    usage: prepare_error_distribution.py [-h] [--results RESULTS]
                                         [--dataset DATASET]
    
    Prepare distribution of mistakes per board for LaTeX
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare distribution of mistakes per board for LaTeX")
    parser.add_argument("--results", help="parent results folder",
                        type=str, default="results://recognition")
    parser.add_argument("--dataset", help="the dataset to evaluate",
                        type=str, default="train")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(URI(args.results) / f"{args.dataset}.csv")

    # Filter out samples where the corners could not be detected
    # df = df[(df["num_incorrect_corners"] != 4) | (df["error"] != "None")]

    counts = df["num_incorrect_squares"].value_counts()
    counts = counts / counts.sum() * 100
    counts = counts[counts.index != 0]
    for i, count in zip(counts.index, counts):
        print(f"({i:2d},{count:5.02f})")

    print(
        f"Proportion of boards classified with >=2 mistakes: {counts[counts.index >= 2].sum():.02}%")
