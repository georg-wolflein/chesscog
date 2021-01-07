"""Utilities for visualizing the functionality of the chessboard corner detector.
"""

import cv2
import numpy as np


def draw_lines(img: np.ndarray, lines: np.ndarray, color: tuple = (0, 0, 255), thickness: int = 2):
    """Draw lines specified in Hough space on top of the input image.

    Args:
        img (np.ndarray): the input image
        lines (np.ndarray): the lines of shape [N, 2] where the last dimension are the rho and theta values
        color (tuple, optional): the color to draw the lines in. Defaults to (0, 0, 255).
        thickness (int, optional): thickness of the lines. Defaults to 2.
    """
    length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
