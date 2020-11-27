import numpy as np
import chess
from pathlib import Path
import torch
from PIL import Image
import functools
import cv2
import argparse
import typing
from recap import URI, CfgNode as CN

from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from chesscog.utils import device, DEVICE
from chesscog.utils.dataset import build_transforms, Datasets
from chesscog.utils.dataset import name_to_piece


class ChessRecognizer:

    _squares = list(chess.SQUARES)

    def __init__(self):
        self._corner_detection_cfg = CN.load_yaml_with_base(
            "config://corner_detection.yaml")

        self._occupancy_cfg, self._occupancy_model = self._load_classifier(
            URI("models://occupancy_classifier"))
        self._occupancy_transforms = build_transforms(
            self._occupancy_cfg, mode=Datasets.TEST)
        self._pieces_cfg, self._pieces_model = self._load_classifier(
            URI("models://piece_classifier"))
        self._pieces_transforms = build_transforms(
            self._pieces_cfg, mode=Datasets.TEST)
        self._piece_classes = np.array(list(map(name_to_piece,
                                                self._pieces_cfg.DATASET.CLASSES)))

    @classmethod
    def _load_classifier(cls, path: Path):
        model_file = next(iter(path.glob("*.pt")))
        yaml_file = next(iter(path.glob("*.yaml")))
        cfg = CN.load_yaml_with_base(yaml_file)
        model = torch.load(model_file, map_location=DEVICE)
        model = device(model)
        model.eval()
        return cfg, model

    def _classify_occupancy(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray) -> np.ndarray:
        warped = create_occupancy_dataset.warp_chessboard_image(
            img, corners)
        square_imgs = map(functools.partial(
            create_occupancy_dataset.crop_square, warped, turn=turn), self._squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy

    def _classify_pieces(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray, occupancy: np.ndarray):
        occupied_squares = np.array(self._squares)[occupancy]
        warped = create_piece_dataset.warp_chessboard_image(
            img, corners)
        piece_imgs = map(functools.partial(
            create_piece_dataset.crop_square, warped, turn=turn), occupied_squares)
        piece_imgs = map(Image.fromarray, piece_imgs)
        piece_imgs = map(self._pieces_transforms, piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs)
        pieces = self._pieces_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        pieces = self._piece_classes[pieces]
        all_pieces = np.full(len(self._squares), None, dtype=np.object)
        all_pieces[occupancy] = pieces
        return all_pieces

    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        with torch.no_grad():
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            occupancy = self._classify_occupancy(img, turn, corners)
            pieces = self._classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            return board, corners


if __name__ == "__main__":
    from chesscog.occupancy_classifier.download_model import ensure_model as ensure_occupancy_classifier
    from chesscog.piece_classifier.download_model import ensure_model as ensure_piece_classifier

    parser = argparse.ArgumentParser(
        description="Run the chess recognition pipeline on an input image")
    parser.add_argument("file", help="path to the input image", type=str)
    parser.add_argument(
        "--white", help="indicate that the image is from the white player's perspective (default)", action="store_true", dest="color")
    parser.add_argument(
        "--black", help="indicate that the image is from the black player's perspective", action="store_false", dest="color")
    parser.set_defaults(color=True)
    args = parser.parse_args()

    ensure_occupancy_classifier(show_size=True)
    ensure_piece_classifier(show_size=True)

    img = cv2.imread(str(URI(args.file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer()
    board, *_ = recognizer.predict(img, args.color)

    print(board)
    print()
    print(
        f"You can view this position at https://lichess.org/editor/{board.board_fen()}")
