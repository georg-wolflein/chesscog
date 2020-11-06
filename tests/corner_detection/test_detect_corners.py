import numpy as np

from chesscog.corner_detection.detect_corners import get_intersection_point


def _get_intersection_point_using_slope_intercept(m1, c1, m2, c2):
    # y = m1*x + c1 = m2*x + c2
    # (m1 - m2)*x = c2 - c1
    # x = (c2 - c1) / (m1 - m2)
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y


def _polar_to_slope_intercept_form(rho, theta):
    m = -np.cos(theta) / np.sin(theta)
    c = rho / np.sin(theta)
    return m, c


def test_get_intersection_point():
    n = 10
    rho1, rho2 = np.random.uniform(1., 10, (2, n))
    theta1, theta2 = np.random.uniform(1., 359., (2, n)) * np.pi / 180
    m1, c1 = _polar_to_slope_intercept_form(rho1, theta1)
    m2, c2 = _polar_to_slope_intercept_form(rho2, theta2)
    expected = _get_intersection_point_using_slope_intercept(m1, c1, m2, c2)
    actual = get_intersection_point(rho1, theta1, rho2, theta2)
    assert np.allclose(expected, actual)
