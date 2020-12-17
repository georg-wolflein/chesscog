import torch
import torchvision
import typing
import logging
from enum import Enum
import numpy as np
import chess
from recap import URI, CfgNode as CN

from .transforms import build_transforms
from . import Datasets

logger = logging.getLogger(__name__)


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


def build_dataset(cfg: CN, mode: Datasets) -> torch.utils.data.Dataset:
    transform = build_transforms(cfg, mode)
    dataset = torchvision.datasets.ImageFolder(root=URI(cfg.DATASET.PATH) / mode.value,
                                               transform=transform)
    return dataset


def build_data_loader(cfg: CN, dataset: torch.utils.data.Dataset, mode: Datasets) -> torch.utils.data.DataLoader:
    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATASET.WORKERS)
