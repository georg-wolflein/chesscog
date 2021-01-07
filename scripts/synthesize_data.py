"""Blender script used to generate the synthetic dataset.
"""

import bpy
import bpy_extras.object_utils
import mathutils
import chess
import numpy as np
from pathlib import Path
import typing
import json
import sys

MIN_BOARD_CORNER_PADDING = 25  # pixels
SQUARE_LENGTH = 0.036
# units (1 unit is the side length of a chessboard square)
CAMERA_DISTANCE = 11
COLLECTION_NAME = "Chess position"


def point_to(obj, focus: mathutils.Vector, roll: float = 0):
    # Based on https://blender.stackexchange.com/a/127440
    loc = obj.location
    direction = focus - loc
    quat = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    roll_matrix = mathutils.Matrix.Rotation(roll, 4, "Z")
    loc = loc.to_tuple()
    obj.matrix_world = quat @ roll_matrix
    obj.location = loc


def setup_camera(color: chess.Color) -> dict:
    camera = bpy.context.scene.camera
    angle = np.random.randint(45, 60)
    z = SQUARE_LENGTH * CAMERA_DISTANCE * np.sin(np.deg2rad(angle))
    y = SQUARE_LENGTH * CAMERA_DISTANCE * np.cos(np.deg2rad(angle))
    y = -y if color == chess.WHITE else y
    x = np.random.normal(0., .8 * SQUARE_LENGTH)
    loc = mathutils.Vector((x, y, z))
    camera.location = loc

    # Update matrix_world as per https://wiki.blender.org/wiki/Reference/Release_Notes/2.80/Python_API/Scene_and_Object_API#Scene_update
    bpy.context.view_layer.update()

    return {
        "angle": angle,
        "location": loc.to_tuple()
    }


def setup_spotlight(light) -> dict:
    angle_xy_plane = np.random.randint(0, 360)
    focus_x, focus_y = np.random.multivariate_normal(
        (0., 0.), np.eye(2) * 2.5 * SQUARE_LENGTH)
    focus = mathutils.Vector((focus_x, focus_y, 0.))
    radius = 10 * SQUARE_LENGTH
    x = radius * np.cos(np.deg2rad(angle_xy_plane))
    y = radius * np.sin(np.deg2rad(angle_xy_plane))
    z = np.random.uniform(5, 10) * SQUARE_LENGTH
    location = mathutils.Vector((x, y, z))
    light.location = location
    point_to(light, focus)
    return {
        "xy_angle": angle_xy_plane,
        "focus": focus.to_tuple(),
        "location": location.to_tuple()
    }


def setup_lighting() -> dict:
    flash = bpy.data.objects["Camera flash light"]
    spot1 = bpy.data.objects["Spot 1"]
    spot2 = bpy.data.objects["Spot 2"]

    modes = {
        "flash": {
            flash: True,
            spot1: False,
            spot2: False
        },
        "spotlights": {
            flash: False,
            spot1: True,
            spot2: True
        }
    }
    mode, visibilities = list(modes.items())[np.random.randint(len(modes))]

    for obj, visibility in visibilities.items():
        obj.hide_render = not visibility

    return {
        "mode": mode,
        "flash": {
            "active": not flash.hide_render
        },
        **{
            key: {
                "active": not obj.hide_render,
                **setup_spotlight(obj)
            } for (key, obj) in {"spot1": spot1, "spot2": spot2}.items()
        }
    }


def add_piece(piece: chess.Piece, square: chess.Square, collection):
    color = {
        chess.WHITE: "White",
        chess.BLACK: "Black"
    }[piece.color]
    piece = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King"
    }[piece.piece_type]
    name = color + " " + piece

    # Position the piece in the middle of the square
    offsets = np.random.normal((.5,)*2, (.1,)*2)
    offsets = np.clip(offsets, .3, .7)
    rank_offset, file_offset = offsets
    rank = chess.square_rank(square) + rank_offset
    file = chess.square_file(square) + file_offset

    # Translate to coordinate system where the origin is in the middle of the board
    rank -= 4
    file -= 4

    location = mathutils.Vector((file, rank, 0)) * SQUARE_LENGTH
    rotation = mathutils.Euler((0., 0., np.random.uniform(0., 360.)))

    src_obj = bpy.data.objects[name]
    obj = src_obj.copy()
    obj.data = src_obj.data.copy()
    obj.animation_data_clear()
    obj.location = location
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    return obj


def render_board(board: chess.Board, turn: chess.Color, output_file: Path):
    scene = bpy.context.scene

    # Setup rendering
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_file)
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 800

    corner_coords = None
    while not corner_coords:
        camera_params = setup_camera(turn)
        lighting_params = setup_lighting()
        corner_coords = get_corner_coordinates(scene)

    # Create a collection to store the position
    if COLLECTION_NAME not in bpy.data.collections:
        collection = bpy.data.collections.new("Chess position")
        scene.collection.children.link(collection)
    collection = bpy.data.collections["Chess position"]

    # Remove all objects from the collection
    bpy.ops.object.delete({"selected_objects": collection.objects})

    piece_data = []
    for square, piece in board.piece_map().items():
        obj = add_piece(piece, square, collection)
        piece_data.append({
            "piece": piece.symbol(),
            "square": chess.square_name(square),
            "box": get_bounding_box(scene, obj)
        })

    # Write data output
    data = {
        "fen": board.board_fen(),
        "white_turn": turn,
        "camera": camera_params,
        "lighting": lighting_params,
        "corners": corner_coords,
        "pieces": piece_data
    }
    with (output_file.parent / (output_file.stem + ".json")).open("w") as f:
        json.dump(data, f)

    # Perform the rendering
    bpy.ops.render.render(write_still=1)


def get_corner_coordinates(scene) -> typing.List[typing.List[int]]:
    corner_points = np.array([[-1., -1],
                              [-1, 1],
                              [1, 1],
                              [1, -1]]) * 4 * SQUARE_LENGTH
    corner_points = np.concatenate((corner_points, np.zeros((4, 1))), axis=-1)
    sr = bpy.context.scene.render

    def _get_coords():
        for corner in corner_points:
            x, y, z = bpy_extras.object_utils.world_to_camera_view(
                scene, scene.camera, mathutils.Vector(corner)).to_tuple()
            y = 1. - y
            x *= sr.resolution_x * sr.resolution_percentage * .01
            y *= sr.resolution_y * sr.resolution_percentage * .01
            x, y = round(x), round(y)

            if not (MIN_BOARD_CORNER_PADDING <= x <= sr.resolution_x - MIN_BOARD_CORNER_PADDING) or \
                    not (MIN_BOARD_CORNER_PADDING <= y <= sr.resolution_y - MIN_BOARD_CORNER_PADDING):
                raise ValueError

            yield x, y
    try:
        return list(_get_coords())
    except ValueError:
        return None


def get_bounding_box(scene, obj) -> typing.Tuple[int, int, int, int]:
    """Obtain the bounding box of an object.

    Args:
        scene: the scene
        obj: the object

    Returns:
        typing.Tuple[int, int, int, int]: the box coordinates in the form (x, y, width, height)
    """
    # adapted from https://blender.stackexchange.com/a/158236
    cam_ob = scene.camera
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = obj.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(obj.matrix_world)
    me.transform(mat)

    camera = cam_ob.data

    def _get_coords():
        frame = [-v for v in camera.view_frame(scene=scene)[:3]]
        for v in me.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                print("===========", z, obj, file=sys.stderr)
                continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            yield x, y

    xs, ys = np.array(list(_get_coords())).T

    min_x = np.clip(min(xs), 0.0, 1.0)
    max_x = np.clip(max(xs), 0.0, 1.0)
    min_y = np.clip(min(ys), 0.0, 1.0)
    max_y = np.clip(max(ys), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    assert round((max_x - min_x) *
                 dim_x) != 0 and round((max_y - min_y) * dim_y) != 0

    return (
        int(round(min_x * dim_x)),
        int(round(dim_y - max_y * dim_y)),
        int(round((max_x - min_x) * dim_x)),
        int(round((max_y - min_y) * dim_y))
    )


if __name__ == "__main__":
    fens_path = Path("fens.txt")
    with fens_path.open("r") as f:
        for i, fen in enumerate(map(str.strip, f)):
            if i <= 2588:
                continue
            print(f"FEN #{i}", file=sys.stderr)
            turn, *fen = fen
            filename = Path("render") / f"{i:04d}.png"
            board = chess.Board("".join(fen))
            render_board(board, turn == "W", filename)
