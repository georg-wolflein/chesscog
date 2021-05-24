"""Script to evaluate the performance of the recognition pipeline.

.. code-block:: console

    $ python -m chesscog.recognition.evaluate --help  
    usage: evaluate.py [-h] [--dataset {train,val,test}] [--out OUT]
                       [--save-fens]
    
    Evaluate the chess recognition system end-to-end.
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset {train,val,test}
                            the dataset to evaluate (if unspecified, train and
                            val will be evaluated)
      --out OUT             output folder
      --save-fens           store predicted and actual FEN strings
"""

import argparse
import typing
from pathlib import Path
from recap import URI
import logging
import json
import cv2
import chess
from chess import Status
import numpy as np
from timeit import default_timer as timer

from .recognition import TimedChessRecognizer
from chesscog.core import sort_corner_points
from chesscog.core.dataset import Datasets
from chesscog.core.exceptions import RecognitionException

logger = logging.getLogger()


def _get_num_mistakes(groundtruth: chess.Board, predicted: chess.Board) -> int:
    groundtruth_map = groundtruth.piece_map()
    predicted_map = predicted.piece_map()
    return sum(0 if groundtruth_map.get(i, None) == predicted_map.get(i, None) else 1
               for i in chess.SQUARES)


def _get_num_occupancy_mistakes(groundtruth: chess.Board, predicted: chess.Board) -> int:
    groundtruth_map = groundtruth.piece_map()
    predicted_map = predicted.piece_map()
    return sum(0 if (i in groundtruth_map) == (i in predicted_map) else 1
               for i in chess.SQUARES)


def _get_num_piece_mistakes(groundtruth: chess.Board, predicted: chess.Board) -> int:
    groundtruth_map = groundtruth.piece_map()
    predicted_map = predicted.piece_map()
    squares = filter(
        lambda i: i in groundtruth_map and i in predicted_map, chess.SQUARES)
    return sum(0 if (groundtruth_map.get(i) == predicted_map.get(i)) else 1
               for i in squares)


def evaluate(recognizer: TimedChessRecognizer, output_file: typing.IO, dataset_folder: Path, save_fens: bool = False):
    """Perform the performance evaluation, saving the results to a CSV output file.

    Args:
        recognizer (TimedChessRecognizer): the instance of the chess recognition pipeline
        output_file (typing.IO): the output file object
        dataset_folder (Path): the folder of the dataset to evaluate
        save_fens (bool, optional): whether to save the FEN outputs for every sample. Defaults to False.
    """

    time_keys = ["corner_detection",
                 "occupancy_classification",
                 "piece_classification",
                 "prepare_results"]
    output_file.write(",".join(["file",
                                "error",
                                "num_incorrect_squares",
                                "num_incorrect_corners",
                                "occupancy_classification_mistakes",
                                "piece_classification_mistakes",
                                "actual_num_pieces",
                                "predicted_num_pieces",
                                *(["fen_actual", "fen_predicted", "fen_predicted_is_valid"]
                                  if save_fens else []),
                                "time_corner_detection",
                                "time_occupancy_classification",
                                "time_piece_classification",
                                "time_prepare_results"]) + "\n")
    for i, img_file in enumerate(dataset_folder.glob("*.png")):
        json_file = img_file.parent / (img_file.stem + ".json")
        with json_file.open("r") as f:
            label = json.load(f)

        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        groundtruth_board = chess.Board(label["fen"])
        groundtruth_corners = sort_corner_points(np.array(label["corners"]))
        error = None
        try:
            predicted_board, predicted_corners, times = recognizer.predict(img,
                                                                           label["white_turn"])
        except RecognitionException as e:
            error = e
            predicted_board = chess.Board()
            predicted_board.clear_board()
            predicted_corners = np.zeros((4, 2))
            times = {k: -1 for k in time_keys}

        mistakes = _get_num_mistakes(groundtruth_board, predicted_board)
        incorrect_corners = np.sum(np.linalg.norm(
            groundtruth_corners - predicted_corners, axis=-1) > (10/1200*img.shape[1]))
        occupancy_mistakes = _get_num_occupancy_mistakes(
            groundtruth_board, predicted_board)
        piece_mistakes = _get_num_piece_mistakes(
            groundtruth_board, predicted_board)

        output_file.write(",".join(map(str, [img_file.name,
                                             error,
                                             mistakes,
                                             incorrect_corners,
                                             occupancy_mistakes,
                                             piece_mistakes,
                                             len(groundtruth_board.piece_map()),
                                             len(predicted_board.piece_map()),
                                             *([groundtruth_board.board_fen(),
                                                predicted_board.board_fen(),
                                                predicted_board.status() == Status.VALID]
                                               if save_fens else []),
                                             *(times[k] for k in time_keys)])) + "\n")
        if (i+1) % 5 == 0:
            output_file.flush()
            logging.info(f"Processed {i+1} files from {dataset_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the chess recognition system end-to-end.")
    parser.add_argument("--dataset", help="the dataset to evaluate (if unspecified, train and val will be evaluated)",
                        type=str, default=None, choices=[x.value for x in Datasets])
    parser.add_argument("--out", help="output folder", type=str,
                        default=f"results://recognition")
    parser.add_argument("--save-fens", help="store predicted and actual FEN strings",
                        action="store_true", dest="save_fens")
    parser.set_defaults(save_fens=False)
    args = parser.parse_args()
    output_folder = URI(args.out)
    output_folder.mkdir(parents=True, exist_ok=True)

    datasets = [Datasets.TRAIN, Datasets.VAL] \
        if args.dataset is None else [d for d in Datasets if d.value == args.dataset]

    recognizer = TimedChessRecognizer()

    for dataset in datasets:
        folder = URI("data://render") / dataset.value
        logger.info(f"Evaluating dataset {folder}")
        with (output_folder / f"{dataset.value}.csv").open("w") as f:
            evaluate(recognizer, f, folder, save_fens=args.save_fens)
