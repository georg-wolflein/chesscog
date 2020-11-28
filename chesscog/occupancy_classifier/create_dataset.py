from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import json
import numpy as np
import chess
import os
import shutil
from recap import URI

from chesscog.core import sort_corner_points

RENDERS_DIR = URI("data://render")
OUT_DIR = URI("data://occupancy")
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE


def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    src_points = sort_corner_points(corners)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=np.float)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))


def extract_squares_from_sample(id: str, subset: str = ""):
    img = cv2.imread(str(RENDERS_DIR / subset / (id + ".png")))
    with (RENDERS_DIR / subset / (id + ".json")).open("r") as f:
        label = json.load(f)

    corners = np.array(label["corners"], dtype=np.float)
    unwarped = warp_chessboard_image(img, corners)

    board = chess.Board(label["fen"])

    for square in chess.SQUARES:
        target_class = "empty" if board.piece_at(
            square) is None else "occupied"
        piece_img = crop_square(unwarped, square, label["white_turn"])
        with Image.fromarray(piece_img, "RGB") as piece_img:
            piece_img.save(OUT_DIR / subset / target_class /
                           f"{id}_{chess.square_name(square)}.png")


if __name__ == "__main__":
    for subset in ("train", "val", "test"):
        for c in ("empty", "occupied"):
            folder = OUT_DIR / subset / c
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
        samples = list((RENDERS_DIR / subset).glob("*.png"))
        for i, img_file in enumerate(samples):
            if i % int(len(samples) / 100) == 0:
                print(f"{i / len(samples)*100:.0f}%")
            extract_squares_from_sample(img_file.stem, subset)
