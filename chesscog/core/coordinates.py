"""Utility functions to convert between Cartesian and homogenous coordinates.
"""

import numpy as np


def to_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Convert Cartesian to homogenous coordinates.

    Args:
        coordinates (np.ndarray): the Cartesian coordinates (shape: [..., 2])

    Returns:
        np.ndarray: the homogenous coordinates (shape: [..., 3])
    """
    return np.concatenate([coordinates,
                           np.ones((*coordinates.shape[:-1], 1))], axis=-1)


def from_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Convert homogenous to Cartesian coordinates.

    Args:
        coordinates (np.ndarray): the homogenous coordinates (shape: [..., 3])

    Returns:
        np.ndarray: the Cartesian coordinates (shape: [..., 2])
    """
    return coordinates[..., :2] / coordinates[..., 2, np.newaxis]
