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

from chesscog.utils import sort_corner_points
from chesscog.utils.dataset import piece_name

RENDERS_DIR = URI("data://render")
OUT_DIR = URI("data://pieces")
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = .25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)


def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    height_increase = MIN_HEIGHT_INCREASE + \
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)
    left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
    right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)
    x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
    y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1))
    width = x2-x1
    height = y2-y1
    cropped_piece = img[y1:y2, x1:x2]
    if col < 4:
        cropped_piece = cv2.flip(cropped_piece, 1)
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped_piece.dtype)
    result[OUT_HEIGHT - height:, :width] = cropped_piece
    return result


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    src_points = sort_corner_points(corners)
    dst_points = np.array([[MARGIN, MARGIN],  # top left
                           [BOARD_SIZE + MARGIN, MARGIN],  # top right
                           [BOARD_SIZE + MARGIN, \
                            BOARD_SIZE + MARGIN],  # bottom right
                           [MARGIN, BOARD_SIZE + MARGIN]  # bottom left
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

    for square, piece in board.piece_map().items():
        piece_img = crop_square(unwarped, square, label["white_turn"])
        with Image.fromarray(piece_img, "RGB") as piece_img:
            piece_img.save(OUT_DIR / subset / piece_name(piece) /
                           f"{id}_{chess.square_name(square)}.png")


def create_folders(subset: str):
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            piece = chess.Piece(piece_type, color)
            folder = OUT_DIR / subset / piece_name(piece)
            folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for subset in ("train", "val", "test"):
        create_folders(subset)
        samples = list((RENDERS_DIR / subset).glob("*.png"))
        for i, img_file in enumerate(samples):
            if i % int(len(samples) / 100) == 0:
                print(f"{i / len(samples)*100:.0f}%")
            extract_squares_from_sample(img_file.stem, subset)
