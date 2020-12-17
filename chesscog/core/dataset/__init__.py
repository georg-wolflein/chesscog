from enum import Enum

from .dataset import color_name, piece_name, name_to_piece, build_transforms, build_dataset, build_data_loader
from .transforms import unnormalize, build_transforms


class Datasets(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
