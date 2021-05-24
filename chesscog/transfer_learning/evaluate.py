"""Script to evaluate the performance of the fine-tuned recognition pipeline on the new chess set.

.. code-block:: console

    $ python -m chesscog.transfer_learning.evaluate --help       
    usage: evaluate.py [-h] [--dataset {train,val,test}] [--out OUT]
    
    Evaluate the chess recognition system end-to-end.
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset {train,val,test}
                            the dataset to evaluate (if unspecified, train and
                            test will be evaluated)
      --out OUT             output folder
"""

import argparse
from recap import URI
import logging

from chesscog.recognition.recognition import TimedChessRecognizer
from chesscog.recognition.evaluate import evaluate
from chesscog.core import sort_corner_points
from chesscog.core.dataset import Datasets
from chesscog.core.exceptions import RecognitionException

logger = logging.getLogger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the chess recognition system end-to-end.")
    parser.add_argument("--dataset", help="the dataset to evaluate (if unspecified, train and test will be evaluated)",
                        type=str, default=None, choices=[x.value for x in Datasets])
    parser.add_argument("--out", help="output folder", type=str,
                        default=f"results://transfer_learning/recognition")
    parser.set_defaults(find_mistakes=False)
    args = parser.parse_args()
    output_folder = URI(args.out)
    output_folder.mkdir(parents=True, exist_ok=True)

    datasets = [Datasets.TRAIN, Datasets.TEST] \
        if args.dataset is None else [d for d in Datasets if d.value == args.dataset]

    recognizer = TimedChessRecognizer(URI("models://transfer_learning"))

    for dataset in datasets:
        folder = URI("data://transfer_learning/images") / dataset.value
        logger.info(f"Evaluating dataset {folder}")
        with (output_folder / f"{dataset.value}.csv").open("w") as f:
            evaluate(recognizer, f, folder, save_fens=True)
