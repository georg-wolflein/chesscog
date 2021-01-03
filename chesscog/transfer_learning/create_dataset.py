"""Script to create the piece and occupancy classification datasets from the input dataset of images.

.. code-block:: console

    $ python -m chesscog.transfer_learning.create_dataset --help  
    usage: create_dataset.py [-h]
    
    Create the piece and occupancy classification datasets.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

from recap import URI, CfgNode as CN
import cv2
import functools
import json
import chess
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.piece_classifier import create_dataset as create_pieces_dataset
from chesscog.core.dataset import Datasets

DATASET_DIR = URI("data://transfer_learning")


def _add_corners_to_train_labels(input_dir: Path):
    corner_detection_cfg = CN.load_yaml_with_base(
        "config://corner_detection.yaml")
    for subset in (x.value for x in (Datasets.TRAIN, Datasets.TEST)):
        for img_file in (input_dir / subset).glob("*.png"):
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, img_scale = resize_image(corner_detection_cfg, img)
            corners = find_corners(corner_detection_cfg, img)
            corners = corners / img_scale

            json_file = img_file.parent / f"{img_file.stem}.json"
            with json_file.open("r") as f:
                label = json.load(f)
            label["corners"] = corners.tolist()
            with json_file.open("w") as f:
                json.dump(label, f)


def _create_dataset(input_dir: Path = DATASET_DIR / "images"):
    _add_corners_to_train_labels(input_dir)
    create_occupancy_dataset.create_dataset(input_dir,
                                            DATASET_DIR / "occupancy")
    create_pieces_dataset.create_dataset(input_dir,
                                         DATASET_DIR / "pieces")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Create the piece and occupancy classification datasets.").parse_args()
    _create_dataset()
