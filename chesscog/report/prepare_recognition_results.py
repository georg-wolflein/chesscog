import pandas as pd
import re
import argparse
from recap import URI

from chesscog.core.dataset import Datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare results for LaTeX")
    parser.add_argument("--dataset", help="the dataset to evaluate",
                        type=str, default="train", choices=[x.value for x in Datasets])
    args = parser.parse_args()

    df = pd.read_csv(URI(f"results://recognition/{args.dataset}.csv"))
    total = len(df)

    # End-to-end accuracy
    num_correct_boards = (df["num_incorrect_squares"] == 0).sum()
    print("End-to-end accuracy:", num_correct_boards / total)

    # End-to-end accuracy allowing one mistake
    num_correct_boards_allowing_one_mistake = (
        df["num_incorrect_squares"] <= 1).sum()
    print("End-to-end accuracy, allowing one mistake:",
          num_correct_boards_allowing_one_mistake / total)

    # Correctly detected corners
    df = df[(df["num_incorrect_corners"] != 4) | (df["error"] != "None")]
    num_correct_corners = len(df)
    print("Corner detection accuracy:", num_correct_corners / total)

    # Correctly classified occupancies
    df_incorrect_occupancies = df[df["actual_num_pieces"]
                                  != df["predicted_num_pieces"]]
    df = df[df["actual_num_pieces"] == df["predicted_num_pieces"]]
    num_correct_occupancies = len(df)
    num_incorrect_occupancies = len(df_incorrect_occupancies)
    print("Occupancy classification accuracy:",
          num_correct_occupancies / num_correct_corners)
    print("  fraction of false positives within occupancy classification errors:", (df_incorrect_occupancies["actual_num_pieces"]
                                                                                    > df_incorrect_occupancies["predicted_num_pieces"]).sum() / num_incorrect_occupancies)

    # Correctly classified pieces
    df = df[df["num_incorrect_squares"] == 0]
    num_correct_pieces = len(df)
    print("Piece classification accuracy:",
          num_correct_pieces / num_correct_occupancies)

    # Performance profiling
    time_cols = [x for x in df.columns if x.startswith("time_")]
    for c in time_cols:
        print(f"Mean {c}:", df[c].mean())

    mean_total_time = df[time_cols].sum(axis=1).mean()
    print("Mean total time:", mean_total_time)
