import numpy as np

from chesscog.core.coordinates import from_homogenous_coordinates, to_homogenous_coordinates


def test_from_homogenous_coordinates():
    coords = np.array([2., 4., 2.])
    expected = np.array([1., 2.])
    assert np.allclose(from_homogenous_coordinates(coords), expected)


def test_to_homogenous_coordinates():
    coords = np.array([1., 2.])
    actual = to_homogenous_coordinates(coords)
    assert actual[2] != 0
    assert np.allclose(from_homogenous_coordinates(actual), coords)
