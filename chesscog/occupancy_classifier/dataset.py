import torch
import torchvision
from torchvision import transforms as T
import typing
import logging
from enum import Enum

from chesscog.utils.config import CfgNode as CN
from chesscog.utils.io import URI

logger = logging.getLogger(__name__)


class Datasets(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def build_transforms(cfg: CN, mode: Datasets) -> typing.Callable:
    transforms = cfg.DATASET.TRANSFORMS
    return T.Compose([
        T.CenterCrop(transforms.CENTER_CROP),
        *([T.RandomHorizontalFlip()] if mode == Datasets.TRAIN else []),
        T.Resize(transforms.RESIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


def build_dataset(cfg: CN, mode: Datasets) -> torch.utils.data.Dataset:
    transform = build_transforms(cfg, mode)
    dataset = torchvision.datasets.ImageFolder(root=URI(cfg.DATASET.PATH) / mode.value,
                                               transform=transform)
    return dataset


def build_data_loader(cfg: CN, dataset: torch.utils.data.Dataset, mode: Datasets) -> torch.utils.data.DataLoader:
    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATASET.WORKERS)
