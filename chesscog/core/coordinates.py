import numpy as np


def to_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    return np.concatenate([coordinates,
                           np.ones((*coordinates.shape[:-1], 1))], axis=-1)


def from_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    return coordinates[..., :2] / coordinates[..., 2, np.newaxis]
