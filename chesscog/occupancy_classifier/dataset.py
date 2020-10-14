import torch
import torchvision
from torchvision import transforms as T
import typing
import logging
from fvcore.common.config import PathManager

from chesscog.config import CfgNode as CN

logger = logging.getLogger(__name__)


def build_transforms(cfg: CN) -> typing.Callable:
    transforms = cfg.DATASET.TRANSFORMS
    return T.Compose([
        T.CenterCrop(transforms.CENTER_CROP),
        T.Resize(transforms.RESIZE),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def build_dataset(cfg: CN):
    transform = build_transforms(cfg)
    dataset = torchvision.datasets.ImageFolder(root=PathManager.get_local_path(cfg.DATASET.PATH),
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.DATASET.BATCH_SIZE,
                                               shuffle=True, num_workers=cfg.DATASET.WORKERS)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.DATASET.BATCH_SIZE,
                                             shuffle=True, num_workers=cfg.DATASET.WORKERS)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.DATASET.BATCH_SIZE,
                                              shuffle=False, num_workers=cfg.DATASET.WORKERS)

    return dataset, train_loader, val_loader, test_loader
