"""Script to compute number of invalid positions (i.e. illegal according to chess rules), among the predicted positions that were wrong.

.. code-block:: console

    $ python3 -m chesscog.report.analyze_misclassified_positions --help
    usage: analyze_misclassified_positions.py [-h] [--results RESULTS] [--dataset DATASET]
    
    Analyse misclassified positions
    
    optional arguments:
      -h, --help         show this help message and exit
      --results RESULTS  parent results folder
      --dataset DATASET  the dataset to evaluate
"""

import pandas as pd
import argparse
import sys
from recap import URI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse misclassified positions")
    parser.add_argument("--results", help="parent results folder",
                        type=str, default="results://recognition")
    parser.add_argument("--dataset", help="the dataset to evaluate",
                        type=str, default="test")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(URI(args.results) / f"{args.dataset}.csv")
    if "fen_predicted" not in df.columns or "fen_actual" not in df.columns or "fen_predicted_is_valid" not in df.columns:
        print(
            "FEN columns not found in the CSV; ensure you export the CSV using --save-fens")
        sys.exit(-1)

    # Evaluate
    missclassified = df["num_incorrect_squares"] > 0
    invalid_predictions = ~df["fen_predicted_is_valid"]
    print("Total number of positions:", missclassified.shape[0])
    print("Number of misclassified positions:", missclassified.sum())
    print("  of which are invalid positions:",
          (missclassified & invalid_predictions).sum())
