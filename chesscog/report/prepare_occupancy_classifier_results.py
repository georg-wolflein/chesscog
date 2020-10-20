import pandas as pd
import re

from chesscog.utils.io import URI

if __name__ == "__main__":
    df = pd.read_csv(URI("results://occupancy_classifier.csv"))
    df["misclassified"] = df["confusion_matrix/0/1"] + df["confusion_matrix/1/0"]
    df["accuracy"] *= 100

    df_train = df[df["dataset"] == "train"] \
        .set_index("run") \
        .drop(columns="dataset") \
        .rename(columns=lambda x: f"train_{x}")
    df_val = df[df["dataset"] == "val"] \
        .set_index("run") \
        .drop(columns="dataset") \
        .rename(columns=lambda x: f"val_{x}")
    df = pd.concat((df_train, df_val), axis="columns")
    df = df.sort_values(["val_accuracy", "train_accuracy"], ascending=False)
    df["centercrop"] = df.index.str.endswith("_centercrop")
    df["context"] = "\\cmark"
    df["context"].loc[df["centercrop"]] = "\\xmark"
    df.index = df.index.str.replace("_centercrop", "")

    regex = re.compile(r"CNN(\d+)_(\d+)Conv_(\d+)Pool_(\d+)FC")
    df.index = df.index.str.replace(
        regex, lambda x: "CNN $(" + ",".join(x.group(i) for i in range(1, 5)) + ")$")

    df = df.set_index("centercrop", append=True)
    df.to_csv(URI("report://data/occupancy_classifier.dat"), sep="\t")
    print(df)
