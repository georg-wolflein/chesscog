from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import json
import numpy as np
import chess
import os

from chesscog import DATA_DIR
from chesscog.util import sort_corner_points

RENDERS_DIR = DATA_DIR / "synthesised" / "render"
OUT_DIR = DATA_DIR / "occupancy"
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE

os.makedirs(OUT_DIR / "empty", exist_ok=True)
os.makedirs(OUT_DIR / "occupied", exist_ok=True)


def crop_square(img, square: chess.Square, turn: chess.Color) -> np.ndarray:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]


def extract_squares_from_sample(id: str):
    img = Image.open(RENDERS_DIR / (id + ".png"))
    with (RENDERS_DIR / (id + ".json")).open("r") as f:
        label = json.load(f)

    src_points = np.array(label["corners"], dtype=np.float)
    src_points = sort_corner_points(src_points)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right,
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=np.float)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    unwarped = cv2.warpPerspective(
        np.array(img), transformation_matrix, (IMG_SIZE, IMG_SIZE))

    board = chess.Board(label["fen"])

    for square in chess.SQUARES:
        target_class = "empty" if board.piece_at(
            square) is None else "occupied"
        piece_img = crop_square(unwarped, square, label["white_turn"])
        piece_img = Image.fromarray(piece_img, "RGB")
        piece_img.save(OUT_DIR / target_class /
                       f"{id}_{chess.square_name(square)}.png")


if __name__ == "__main__":
    samples = list(RENDERS_DIR.glob("*.png"))
    for i, img_file in enumerate(samples):
        if i % int(len(samples) / 100) == 0:
            print(f"{i / len(samples)*100:.0f}%")
        extract_squares_from_sample(img_file.stem)
