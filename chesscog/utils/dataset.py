import torch
import torchvision
from torchvision import transforms as T
import typing
import logging
from enum import Enum
import numpy as np
import chess
from recap import URI, CfgNode as CN


logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def color_name(color: chess.Color):
    return {chess.WHITE: "white",
            chess.BLACK: "black"}[color]


def piece_name(piece: chess.Piece) -> str:
    return f"{color_name(piece.color)}_{chess.piece_name(piece.piece_type)}"


def name_to_piece(name: str) -> chess.Piece:
    color, piece_type = name.split("_")
    color = color == "white"
    piece_type = chess.PIECE_NAMES.index(piece_type)
    return chess.Piece(piece_type, color)


class Datasets(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def build_transforms(cfg: CN, mode: Datasets) -> typing.Callable:
    transforms = cfg.DATASET.TRANSFORMS
    t = []
    if transforms.CENTER_CROP:
        t.append(T.CenterCrop(transforms.CENTER_CROP))
    if mode == Datasets.TRAIN and transforms.RANDOM_HORIZONTAL_FLIP:
        t.append(T.RandomHorizontalFlip(transforms.RANDOM_HORIZONTAL_FLIP))
    if transforms.RESIZE:
        t.append(T.Resize(tuple(reversed(transforms.RESIZE))))
    t.extend([T.ToTensor(),
              T.Normalize(mean=_MEAN, std=_STD)])
    return T.Compose(t)


def unnormalize(x: typing.Union[torch.Tensor, np.ndarray]) -> typing.Union[torch.Tensor, np.ndarray]:
    # x must be of the form ([..., W, H, 3])
    return x * _STD + _MEAN


def build_dataset(cfg: CN, mode: Datasets) -> torch.utils.data.Dataset:
    transform = build_transforms(cfg, mode)
    dataset = torchvision.datasets.ImageFolder(root=URI(cfg.DATASET.PATH) / mode.value,
                                               transform=transform)
    return dataset


def build_data_loader(cfg: CN, dataset: torch.utils.data.Dataset, mode: Datasets) -> torch.utils.data.DataLoader:
    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATASET.WORKERS)
