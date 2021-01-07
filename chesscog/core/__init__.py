import numpy as np
import torch
import typing
import functools
from collections.abc import Iterable

#: Device to be used for computation (GPU if available, else CPU).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, torch.nn.Module, typing.List[torch.Tensor],
                 tuple, dict, typing.Generator]


def device(x: T, dev: str = DEVICE) -> T:
    """Convenience method to move a tensor/module/other structure containing tensors to the device.

    Args:
        x (T): the tensor (or strucure containing tensors)
        dev (str, optional): the device to move the tensor to. Defaults to DEVICE.

    Raises:
        TypeError: if the type was not a compatible tensor

    Returns:
        T: the input tensor moved to the device
    """

    to = functools.partial(device, dev=dev)
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        return x.to(dev)
    elif isinstance(x, list):
        return list(map(to, x))
    elif isinstance(x, tuple):
        return tuple(map(to, x))
    elif isinstance(x, dict):
        return {k: to(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return map(to, x)
    else:
        raise TypeError


def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Permute the board corner coordinates to the order [top left, top right, bottom right, bottom left].

    Args:
        points (np.ndarray): the four corner coordinates

    Returns:
        np.ndarray: the permuted array
    """

    # First, order by y-coordinate
    points = points[points[:, 1].argsort()]
    # Sort top x-coordinates
    points[:2] = points[:2][points[:2, 0].argsort()]
    # Sort bottom x-coordinates (reversed)
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]

    return points


def listify(func: typing.Callable[..., typing.Iterable]) -> typing.Callable[..., typing.List]:
    """Decorator to convert the output of a generator function to a list.

    Args:
        func (typing.Callable[..., typing.Iterable]): the function to be decorated

    Returns:
        typing.Callable[..., typing.List]: the decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper
