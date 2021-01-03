from enum import Enum


class Datasets(Enum):
    """Enumeration of the dataset split.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
