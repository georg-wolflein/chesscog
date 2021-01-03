"""Script to visualize the image and labels for a sample from the dataset.

.. code-block:: console

    $ python -m chesscog.data_synthesis.visualize --help    
    usage: visualize.py [-h] [--file FILE]
    
    Visualize a sample from the dataset.
    
    optional arguments:
      -h, --help   show this help message and exit
      --file FILE  path to image file
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import typing
import json
from recap import URI
import argparse


def _draw_board_edges(img: Image, corners: typing.List[typing.List[int]]):
    draw = ImageDraw.Draw(img)
    corners = list(map(tuple, corners))
    corners.append(corners[0])
    draw.line(corners, "red", width=3)


def _draw_bounding_boxes(img: Image, pieces: list):
    try:
        font = ImageFont.truetype('arial.ttf', 50)
    except IOError:
        font = ImageFont.load_default()

    # First, draw the bounding boxes
    labels_to_draw = list()
    for piece in pieces[::-1]:
        draw = ImageDraw.Draw(img)
        name = piece["piece"]
        box = (x, y, w, h) = piece["box"]

        color = "white" if name.isupper() else "black"
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)

        labels_to_draw.append((name, box, color))

    # Draw each bounding box's label in the top left
    for name, (x, y, w, h), outline_color in labels_to_draw:
        draw = ImageDraw.Draw(img)
        text_color = {
            "black": "white",
            "white": "black"
        }[outline_color]

        text = {
            "P": "pawn",
            "N": "knight",
            "B": "bishop",
            "R": "rook",
            "Q": "queen",
            "K": "king"
        }[name.upper()]

        text_width, text_height = font.getsize(text)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            (x,
             y,
             x + text_width + 2 * margin,
             y + text_height + 2 * margin),
            fill=outline_color
        )
        draw.text(
            (x + margin, y + margin),
            text,
            fill=text_color,
            font=font)


def _visualize_groundtruth(img: Image, label: dict):
    _draw_board_edges(img, label["corners"])
    _draw_bounding_boxes(img, label["pieces"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a sample from the dataset.")
    parser.add_argument("--file", type=str, help="path to image file",
                        default="data://render/train/3828.png")
    args = parser.parse_args()

    img_file = URI(args.file)
    json_file = img_file.parent / f"{img_file.stem}.json"

    img = Image.open(img_file)
    with json_file.open("r") as f:
        label = json.load(f)

    _visualize_groundtruth(img, label)
    img.show()
