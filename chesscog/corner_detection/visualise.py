import cv2
import numpy as np


def draw_lines(img: np.ndarray, lines: np.ndarray):
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
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
