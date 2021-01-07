"""Script to split the rendered dataset into train (90%), val (3%), and test (7%) sets.

.. code-block:: console

    $ python -m chesscog.data_synthesis.split_dataset --help
    usage: split_dataset.py [-h]
    
    Split the dataset into train/val/test.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

import numpy as np
from logging import getLogger
from recap import URI
import argparse


logger = getLogger(__name__)

if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Split the dataset into train/val/test.").parse_args()

    val_split = .03
    test_split = .1
    render_dir = URI("data://render")
    ids = np.array([x.stem for x in render_dir.glob("*.json")])
    if len(ids) == 0:
        logger.warning(
            "No samples found in 'data://render', either you did not download the datset yet or you have already split it.")
    np.random.seed(42)
    ids = np.random.permutation(ids)
    sample_sizes = (np.array([val_split, test_split])
                    * len(ids)).astype(np.int32)
    val, test, train = np.split(ids, sample_sizes)
    datasets = {"val": val, "test": test, "train": train}
    print(f"{len(ids)} samples will be split into {len(train)} train, {len(val)} val, {len(test)} test.")

    for dataset_name, ids in datasets.items():
        dataset_dir = render_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        for id in ids:
            (render_dir / f"{id}.png").rename(dataset_dir / f"{id}.png")
            (render_dir / f"{id}.json").rename(dataset_dir / f"{id}.json")
