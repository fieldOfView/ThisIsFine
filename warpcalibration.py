import pyglet
import cv2
import numpy as np
import copy
import json

identity = pyglet.math.Mat4()
matrix = copy.deepcopy(identity)

win = pyglet.window.Window(width=1920, height=1080, fullscreen=True)

src_points = [
    [0, 0],
    [win.width, 0],
    [win.width, win.height],
    [0, win.height]
]
dest_points = copy.deepcopy(src_points)
active_point_index = 0

point_batch = pyglet.graphics.Batch()
point_shapes = [
    pyglet.shapes.Arc(p[0], p[1], radius=8, color=(0, 255, 255), batch=point_batch)
    for p in dest_points
]


def make_matrix():
    homography, mask = cv2.findHomography(np.asarray(src_points), np.asarray(dest_points))
    matrix = pyglet.math.Mat4([
        homography[0][0], homography[1][0], 0.0, homography[2][0],
        homography[0][1], homography[1][1], 0.0, homography[2][1],
        0.0,              0.0,              1.0, 0.0,
        homography[0][2], homography[1][2], 0.0, homography[2][2],
    ])
    return matrix


matrix = make_matrix()

batch = pyglet.graphics.Batch()
rect = pyglet.shapes.Rectangle(0, 0, win.width, win.height, color=(127, 127, 127), batch=batch)
line1 = pyglet.shapes.Line(0, 0, win.width, win.height, batch=batch)
line2 = pyglet.shapes.Line(0, win.height, win.width, 0, batch=batch)
circle = pyglet.shapes.Arc(0.1 * win.width, 0.1 * win.height, radius=0.05 * win.width, color=(0, 255, 0), batch=batch)


@win.event
def on_draw():
    win.clear()
    win.view = matrix
    batch.draw()
    win.view = identity
    point_batch.draw()


@win.event
def on_key_press(symbol, modifiers):
    global active_point_index, matrix, dest_points
    if symbol >= pyglet.window.key._1 and symbol <= pyglet.window.key._4:
        point_shapes[active_point_index].color = (0, 255, 255)
        active_point_index = symbol - pyglet.window.key._1
        point_shapes[active_point_index].color = (255, 255, 0)

    elif symbol == pyglet.window.key.R:
        dest_points = copy.deepcopy(src_points)
        for index, shape in enumerate(point_shapes):
            shape.x = dest_points[index][0]
            shape.y = dest_points[index][1]
        matrix = make_matrix()

    elif symbol == pyglet.window.key.S:
        data = {
            "points": dest_points,
            "matrix": matrix
        }
        with open("resources/warp.json", "w") as f:
            json.dump(data, f)

    elif symbol == pyglet.window.key.L:
        with open("resources/warp.json", "r") as f:
            data = json.load(f)

            dest_points = data["points"]

        for index, shape in enumerate(point_shapes):
            shape.x = dest_points[index][0]
            shape.y = dest_points[index][1]
        matrix = make_matrix()


@win.event
def on_text_motion(motion):
    global active_point_index, matrix
    direction = motion - pyglet.window.key.MOTION_LEFT
    distance = (-1 if (direction >> 1) else 1) * (1 if (direction & 1) else -1)
    dest_points[active_point_index][direction & 1] += distance
    point_shapes[active_point_index].x = dest_points[active_point_index][0]
    point_shapes[active_point_index].y = dest_points[active_point_index][1]

    matrix = make_matrix()


pyglet.app.run()
