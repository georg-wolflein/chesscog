"""Script to prepare the results of a (occupancy or piece) classifier for LaTeX.

.. code-block:: console

    $ python -m chesscog.report.prepare_classifier_results --help
    usage: prepare_classifier_results.py [-h]
                                         [--classifier {occupancy_classifier,piece_classifier}]
    
    Prepare results for LaTeX
    
    optional arguments:
      -h, --help            show this help message and exit
      --classifier {occupancy_classifier,piece_classifier}
"""

import pandas as pd
import re
import argparse
from recap import URI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare results for LaTeX")
    parser.add_argument("--classifier", type=str, choices=[
                        "occupancy_classifier", "piece_classifier"], default="occupancy_classifier")

    args = parser.parse_args()
    classifier = args.classifier

    df = pd.read_csv(URI(f"results://{classifier}/evaluate.csv"))
    pattern = re.compile(r"^confusion_matrix/(\d+)/(\d+)$")
    off_diagonal_confusion_matrix_mask = df.columns \
        .map(pattern.match) \
        .map(lambda match: match and match.group(1) != match.group(2)) \
        .fillna(False)
    df["misclassified"] = df.loc[:,
                                 off_diagonal_confusion_matrix_mask].sum(axis="columns")

    df["accuracy"] *= 100

    df_train = df[df["dataset"] == "train"] \
        .set_index("model") \
        .drop(columns="dataset") \
        .rename(columns=lambda x: f"train_{x}")
    df_val = df[df["dataset"] == "val"] \
        .set_index("model") \
        .drop(columns="dataset") \
        .rename(columns=lambda x: f"val_{x}")
    df = pd.concat((df_train, df_val), axis="columns")
    df = df.sort_values(["val_accuracy", "train_accuracy"], ascending=False)
    df["centercrop"] = df.index.str.endswith("_centercrop")
    df["context"] = "\\cmark"
    df.loc[df["centercrop"], "context"] = "\\xmark"
    df.index = df.index.str.replace("_centercrop", "")
    df["parameters"] = df["val_parameters"]
    df.drop(columns=["train_parameters", "val_parameters"])

    regex = re.compile(r"CNN(\d+)_(\d+)Conv_(\d+)Pool_(\d+)FC")
    df.index = df.index.str.replace(
        regex, lambda x: "\\acs{cnn} $(" + ",".join(x.group(i) for i in range(1, 5)) + ")$")

    citekeys = {
        "InceptionV3": "szegedy2016",
        "VGG": "simonyan2015",
        "ResNet": "he2016",
        "AlexNet": "krizhevsky2017"
    }
    df.index = df.index.map(
        lambda x: f"{x} \\cite{{{citekeys[x]}}}" if x in citekeys else x)

    df = df.set_index("centercrop", append=True)
    df.to_csv(URI(f"report://data/{classifier}.dat"), sep="\t")
    print(df)
