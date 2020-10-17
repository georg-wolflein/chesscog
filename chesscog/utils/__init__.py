import numpy as np
import torch
import typing
import functools

_device = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, typing.List[torch.Tensor], tuple, dict]


def device(x: T, dev: str = _device) -> T:
    to = functools.partial(device, dev=dev)
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    elif isinstance(x, list):
        return list(map(to, x))
    elif isinstance(x, tuple):
        return tuple(map(to, x))
    elif isinstance(x, dict):
        return {k: to(v) for k, v in x.items()}
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
