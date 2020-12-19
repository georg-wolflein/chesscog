import pandas as pd
import re
import argparse
from recap import URI

from chesscog.core.dataset import Datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare results for LaTeX")
    parser.add_argument("--results", help="parent results folder",
                        type=str, default="results://recognition")
    parser.add_argument("--dataset", help="the dataset to evaluate",
                        type=str, default="train", choices=[x.value for x in Datasets])
    args = parser.parse_args()

    df = pd.read_csv(URI(args.results) / f"{args.dataset}.csv")
    total = len(df)

    # End-to-end accuracy
    num_correct_boards = (df["num_incorrect_squares"] == 0).sum()
    print("End-to-end accuracy:", num_correct_boards / total)

    # End-to-end accuracy allowing one mistake
    num_correct_boards_allowing_one_mistake = (
        df["num_incorrect_squares"] <= 1).sum()
    print("End-to-end accuracy, allowing one mistake:",
          num_correct_boards_allowing_one_mistake / total)

    # Mean misclassified
    mean_misclassified = df["num_incorrect_squares"].mean()
    print("Mean number of incorrect squares:", mean_misclassified)

    # Correctly detected corners
    df = df[(df["num_incorrect_corners"] != 4) | (df["error"] != "None")]
    num_correct_corners = len(df)
    print("Corner detection accuracy:", num_correct_corners / total)

    # Occupancy classification
    num_squares = 64 * len(df)
    num_occupancy_mistakes = df["occupancy_classification_mistakes"].sum()
    print("Per-square occupancy classification accuracy:",
          num_occupancy_mistakes / num_squares)

    # Piece classification
    num_occupancy_correct = num_squares - num_occupancy_mistakes
    num_piece_mistakes = df["piece_classification_mistakes"].sum()
    print("Per-square piece classification accuracy:",
          num_piece_mistakes / num_occupancy_correct)

    # Performance profiling
    time_cols = [x for x in df.columns if x.startswith("time_")]
    for c in time_cols:
        print(f"Mean {c}:", df[c].mean())

    mean_total_time = df[time_cols].sum(axis=1).mean()
    print("Mean total time:", mean_total_time)
