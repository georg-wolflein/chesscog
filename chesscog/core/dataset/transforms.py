from recap import CfgNode as CN
import typing
from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image, ImageOps
from abc import ABC

from .datasets import Datasets

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(x: typing.Union[torch.Tensor, np.ndarray]) -> typing.Union[torch.Tensor, np.ndarray]:
    """Unnormalize an input image. 

    It must be of the form ([..., W, H, 3]).

    Args:
        x (typing.Union[torch.Tensor, np.ndarray]): the input tensor/array representing the image

    Returns:
        typing.Union[torch.Tensor, np.ndarray]: the unnormalized image
    """
    return x * _STD + _MEAN


class Shear:
    """Custom shear transform that keeps the bottom of the image invariant because for piece classification, we only want to "tilt" the top of the image.
    """

    def __init__(self, amount: typing.Union[tuple, float, int, None]):
        self.amount = amount

    @classmethod
    def _shear(cls, img: Image, amount: float) -> Image:
        img = ImageOps.flip(img)
        img = img.transform(img.size, Image.AFFINE,
                            (1, -amount, 0, 0, 1, 0))
        img = ImageOps.flip(img)
        return img

    def __call__(self, img: Image) -> Image:
        if not self.amount:
            return img
        if isinstance(self.amount, (tuple, list)):
            min_val, max_val = sorted(self.amount)
        else:
            min_val = max_val = self.amount

        amount = np.random.uniform(low=min_val, high=max_val)
        return self._shear(img, amount)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.amount})"


class _HVTransform(ABC):
    """Base class for transforms parameterized by horizontal and vertical values.
    """

    def __init__(self, horizontal: typing.Union[float, tuple, None], vertical: typing.Union[float, tuple, None]):
        self.horizontal = self._get_tuple(horizontal)
        self.vertical = self._get_tuple(vertical)

    _default_value = None

    @classmethod
    def _get_tuple(cls, value: typing.Union[float, tuple, None]) -> tuple:
        if value is None:
            return cls._default_value, cls._default_value
        elif isinstance(value, (tuple, list)):
            return tuple(map(float, value))
        elif isinstance(value, (float, int)):
            return tuple(map(float, (value, value)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.horizontal}, {self.vertical})"


class Scale(_HVTransform):
    """Custom scaling transform where the horizontal and vertical scales can independently be specified.

    The center of scaling is the bottom left of the image (this makes particular sense for the piece classifier).
    """

    _default_value = 1.

    def __call__(self, img: Image) -> Image:
        w, h = img.size
        w_scale = np.random.uniform(*self.horizontal)
        h_scale = np.random.uniform(*self.vertical)
        w_, h_ = map(int, (w*w_scale, h*h_scale))
        img = img.resize((w_, h_))
        img = img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, h_-h))
        return img


class Translate(_HVTransform):
    """Custom translation transform for convenience.
    """

    _default_value = 0.

    def __call__(self, img: Image) -> Image:
        w, h = img.size
        w_translate = np.random.uniform(*self.horizontal)
        h_translate = np.random.uniform(*self.vertical)
        w_, h_ = map(int, (w*w_translate, h*h_translate))
        img = img.transform((w, h), Image.AFFINE, (1, 0, -w_, 0, 1, h_))
        return img


def build_transforms(cfg: CN, mode: Datasets) -> typing.Callable:
    """Build the transforms for a dataset.

    Args:
        cfg (CN): the config object
        mode (Datasets): the dataset split

    Returns:
        typing.Callable: the transform function
    """
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
        t.append(Shear(transforms.SHEAR))
        t.append(Scale(transforms.SCALE.HORIZONTAL,
                       transforms.SCALE.VERTICAL))
        t.append(Translate(transforms.TRANSLATE.HORIZONTAL,
                           transforms.TRANSLATE.VERTICAL))
    if transforms.RESIZE:
        t.append(T.Resize(tuple(reversed(transforms.RESIZE))))
    t.extend([T.ToTensor(),
              T.Normalize(mean=_MEAN, std=_STD)])
    return T.Compose(t)
