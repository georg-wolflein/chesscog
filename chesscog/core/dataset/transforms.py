from recap import CfgNode as CN
import typing
from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image, ImageTransform, ImageOps

from .datasets import Datasets

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(x: typing.Union[torch.Tensor, np.ndarray]) -> typing.Union[torch.Tensor, np.ndarray]:
    # x must be of the form ([..., W, H, 3])
    return x * _STD + _MEAN


class Shear:
    def __init__(self, amount: typing.Union[tuple, float, int, None]):
        self.amount = amount

    @classmethod
    def _shear(cls, img: Image, amount: float) -> Image:
        transform = ImageTransform.AffineTransform((1, -amount, 0,
                                                    0, 1, 0))
        img = ImageOps.flip(img)
        img = transform.transform(img.size, img)
        img = ImageOps.flip(img)

    def __call__(self, img: Image) -> Image:
        if not self.amount:
            return img
        if isinstance(self.amount, tuple):
            min_val, max_val = sorted(self.amount)
        else:
            min_val = max_val = self.amount

        amount = np.random.uniform(low=min_val, high=max_val)
        return self._shear(img, amount)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.amount})"


def build_transforms(cfg: CN, mode: Datasets) -> typing.Callable:
    transforms = cfg.DATASET.TRANSFORMS
    t = []
    if transforms.CENTER_CROP:
        t.append(T.CenterCrop(transforms.CENTER_CROP))
    if mode == Datasets.TRAIN:
        if transforms.RANDOM_HORIZONTAL_FLIP:
            t.append(T.RandomHorizontalFlip(transforms.RANDOM_HORIZONTAL_FLIP))
        t.append(T.ColorJitter(brightness=transforms.COLOR_JITTER.BRIGHTNESS,
                               contrast=transforms.COLOR_JITTER.CONTRAST,
                               saturation=transforms.COLOR_JITTER.SATURATION,
                               hue=transforms.COLOR_JITTER.HUE))
        shear = transforms.SHEAR
        if shear:
            if isinstance(shear, list):
                shear = tuple(shear)
            t.append(Shear(shear))
    if transforms.RESIZE:
        t.append(T.Resize(tuple(reversed(transforms.RESIZE))))
    t.extend([T.ToTensor(),
              T.Normalize(mean=_MEAN, std=_STD)])
    return T.Compose(t)
