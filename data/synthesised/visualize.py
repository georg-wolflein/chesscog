import numpy as np
from PIL import Image, ImageDraw, ImageFont
import typing


def draw_board_edges(img: Image, corners: typing.List[typing.List[int]]):
    draw = ImageDraw.Draw(img)
    corners = list(map(tuple, corners))
    corners.append(corners[0])
    draw.line(corners, "red", width=3)


def draw_bounding_boxes(img: Image, pieces: list):
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


def visualise_groundtruth(img: Image, label: dict):
    draw_board_edges(img, label["corners"])
    draw_bounding_boxes(img, label["pieces"])


if __name__ == "__main__":
    import json
    from pathlib import Path

    start = 0
    for i in range(start, start+5):
        id = f"{i:04d}"
        dataset_dir = Path("render")

        img = Image.open(dataset_dir / (id + ".png"))
        with (dataset_dir / (id + ".json")).open("r") as f:
            label = json.load(f)

        visualise_groundtruth(img, label)
        img.show()
