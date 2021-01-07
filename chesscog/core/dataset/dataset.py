"""Methods specific to handling chess datasets.
"""

import torch
import torchvision
import typing
import logging
from enum import Enum
import numpy as np
import chess
from recap import URI, CfgNode as CN

from .transforms import build_transforms
from .datasets import Datasets

logger = logging.getLogger(__name__)


def color_name(color: chess.Color) -> str:
    """Convert a chess color to a string.

    Args:
        color (chess.Color): the color

    Returns:
        str: the string representation
    """
    return {chess.WHITE: "white",
            chess.BLACK: "black"}[color]


def piece_name(piece: chess.Piece) -> str:
    """Convert a chess piece to a string.

    Args:
        piece (chess.Piece): the piece

    Returns:
        str: the corresponding string
    """
    return f"{color_name(piece.color)}_{chess.piece_name(piece.piece_type)}"


def name_to_piece(name: str) -> chess.Piece:
    """Convert the name of a piece to an instance of :class:`chess.Piece`.

    Args:
        name (str): the name of the piece

    Returns:
        chess.Piece: the instance of :class:`chess.Piece`
    """
    color, piece_type = name.split("_")
    color = color == "white"
    piece_type = chess.PIECE_NAMES.index(piece_type)
    return chess.Piece(piece_type, color)


def build_dataset(cfg: CN, mode: Datasets) -> torch.utils.data.Dataset:
    """Build a dataset from its configuration.

    Args:
        cfg (CN): the config object
        mode (Datasets): the split (important to figure out which transforms to apply)

    Returns:
        torch.utils.data.Dataset: the dataset
    """
    transform = build_transforms(cfg, mode)
    dataset = torchvision.datasets.ImageFolder(root=URI(cfg.DATASET.PATH) / mode.value,
                                               transform=transform)
    return dataset


def build_data_loader(cfg: CN, dataset: torch.utils.data.Dataset, mode: Datasets) -> torch.utils.data.DataLoader:
    """Build a data loader for a dataset.

    Args:
        cfg (CN): the config object
        dataset (torch.utils.data.Dataset): the dataset
        mode (Datasets): the split

    Returns:
        torch.utils.data.DataLoader: the data loader
    """
    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATASET.WORKERS)
