import argparse
import typing
from pathlib import Path
import cv2
import json
import numpy as np

from chesscog.utils.dataset import Datasets
from chesscog.utils.io import URI
from chesscog.utils import sort_corner_points
from chesscog.corner_detection import find_corners


def evaluate(datasets: typing.List[Datasets], output_folder: Path, find_mistakes: bool = False, include_heading: bool = False) -> str:
    for dataset in datasets:
        mistakes = 0
        total = 0
        folder = URI("data://render") / dataset.value
        for img_file in folder.glob("*.png"):
            total += 1
            img = cv2.imread(str(img_file))
            json_file = folder / f"{img_file.stem}.json"
            with json_file.open("r") as f:
                label = json.load(f)
            actual = np.array(label["corners"])
            predicted = find_corners(img)

            actual = sort_corner_points(actual)
            predicted = sort_corner_points(predicted)

            if np.linalg.norm(actual - predicted, axis=-1).max() > 10.:
                mistakes += 1
                print(mistakes, total)
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(img)
                # plt.scatter(*actual.T, c="g")
                # plt.scatter(*predicted.T, c="r")
                # plt.show()
        print(mistakes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the chessboard corner detector.")
    parser.add_argument("--dataset", help="the dataset to evaluate (if unspecified, train and val will be evaluated)",
                        type=str, default=None, choices=[x.value for x in Datasets])
    parser.add_argument("--out", help="output folder", type=str,
                        default=f"results://corner_detector")
    parser.add_argument("--find-mistakes", help="whether to output all incorrectly detected images",
                        dest="find_mistakes", action="store_true")
    parser.set_defaults(find_mistakes=False)
    args = parser.parse_args()

    datasets = [Datasets.TRAIN, Datasets.VAL] \
        if args.dataset is None else [d for d in Datasets if d.value == args.dataset]
    evaluate(datasets, args.out, args.find_mistakes)
