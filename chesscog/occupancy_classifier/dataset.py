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
    ALL = "all"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


DatasetsDict = typing.Dict[Datasets, torch.utils.data.Dataset]


def build_transforms(cfg: CN) -> typing.Callable:
    transforms = cfg.DATASET.TRANSFORMS
    return T.Compose([
        T.CenterCrop(transforms.CENTER_CROP),
        T.Resize(transforms.RESIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


def build_datasets(cfg: CN) -> typing.Tuple[DatasetsDict, typing.List[str]]:
    transform = build_transforms(cfg)
    dataset = torchvision.datasets.ImageFolder(root=URI(cfg.DATASET.PATH),
                                               transform=transform)
    n_total = len(dataset)
    n_val = int(cfg.DATASET.SPLIT.VAL * n_total)
    n_test = int(cfg.DATASET.SPLIT.TEST * n_total)
    n_train = n_total - n_val - n_test
    logger.info(
        f"Creating dataset with {n_total} samples total, split ({n_train}/{n_val}/{n_test}) into (train/val/test))")

    train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                                 (n_train,
                                                                  n_val, n_test),
                                                                 generator=torch.Generator().manual_seed(42))

    return {
        Datasets.ALL: dataset,
        Datasets.TRAIN: train_set,
        Datasets.VAL: val_set,
        Datasets.TEST: test_set
    }, dataset.classes


def build_data_loader(cfg: CN, datasets: DatasetsDict, mode: Datasets) -> torch.utils.data.DataLoader:
    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(datasets[mode], batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATASET.WORKERS)
