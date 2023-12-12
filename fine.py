import pyglet
import pyglet.gl as gl

import json
import time
import math
import threading
from enum import Enum

from FineServer import FineServer
from FineDetector import FineDetector


class FineState(Enum):
    SHOWING_RESULT = 0
    DETECTING_POSES = 1
    GENERATING_IMAGE = 2


class ThisIsFineWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(caption="This is Fine!", width=1920, height=1080)

        # load quad warp matrix from warp.json generated by warpcalibration.py
        with open("resources/warp.json", "r") as f:
            data = json.load(f)
            self.view = pyglet.math.Mat4(data["matrix"])

        self._generated_image_lock = threading.Lock()
        self._generated_image = pyglet.image.create(960, 540)
        self._pose_image = pyglet.image.create(960, 540)
        self._frozen_pose_image = pyglet.image.create(960, 540)
        self._frozen_pose_png_data = None

        self._flash_batch = pyglet.graphics.Batch()
        self._flash_rect = pyglet.shapes.Rectangle(0, 0, self.width, self.height, batch=self._flash_batch)

        self._countdown_batch = pyglet.graphics.Batch()
        self._countdown_arc = pyglet.shapes.Sector(
            int(self.width / 2),
            int(self.height / 2),
            int(self.height / 4),
            angle=0,
            start_angle=math.pi / 2,
            batch=self._countdown_batch,
        )

        self._wait_batch = pyglet.graphics.Batch()
        self._wait_circles = [
            pyglet.shapes.Circle(
                int(self.width / 2 + c * self.width / 8),
                int(self.height / 2),
                int(self.height / 16),
                batch=self._wait_batch,
            )
            for c in range(-1, 2)
        ]

        self._pose_detector = FineDetector()
        self._server = FineServer()
        self._server.pose_callback = self.getFrozenPoseImagePNG
        self._server.invoked_callback = self.setInvokedImage

        self._state = FineState.SHOWING_RESULT
        self._state_start = time.time()

    def cleanup(self):
        self._server.requestStop()
        self._pose_detector.requestStop()

    def setState(self, state):
        self._state = state
        self._state_start = time.time()

    def on_close(self):
        self.cleanup()
        super().on_close()

    def on_draw(self):
        self.clear()

        time_passed = time.time() - self._state_start

        match self._state:
            case FineState.SHOWING_RESULT:
                with self._generated_image_lock:
                    self._generated_image.blit(0, 0, width=self.width, height=self.height)

                if time_passed >= 10:
                    self.setState(FineState.DETECTING_POSES)
                elif time_passed > 8:
                    self._pose_image = self.getPoseImage()
                    pose_sprite = pyglet.sprite.Sprite(
                        self._pose_image,
                        blend_dest=gl.GL_ONE_MINUS_SRC_COLOR,
                    )
                    pose_sprite.scale_x = self.width / self._pose_image.width
                    pose_sprite.scale_y = self.height / self._pose_image.height
                    pose_sprite.opacity = int(255 * (time_passed - 8) / 2)
                    pose_sprite.draw()
                    pose_sprite.delete()
                elif time_passed < 2:
                    pose_sprite = pyglet.sprite.Sprite(
                        self._frozen_pose_image,
                        blend_dest=gl.GL_ONE_MINUS_SRC_COLOR,
                    )
                    pose_sprite.scale_x = self.width / self._frozen_pose_image.width
                    pose_sprite.scale_y = self.height / self._frozen_pose_image.height
                    pose_sprite.opacity = int(255 * (1 - time_passed / 2))
                    pose_sprite.draw()
                    pose_sprite.delete()

            case FineState.DETECTING_POSES:
                with self._generated_image_lock:
                    self._generated_image.blit(0, 0, width=self.width, height=self.height)

                self._pose_image = self.getPoseImage()
 
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_COLOR)
                self._pose_image.blit(0, 0, width=self.width, height=self.height)
                gl.glDisable(gl.GL_BLEND)

                '''
                if not self._pose_detector.getFullPoseDetected():
                    self.setState(FineState.DETECTING_POSES)
                else:
                    if time_passed > 8:
                        self.invokeGeneration()
                    elif time_passed > 5:
                        progress = (time_passed - 5) / 3
                        self._countdown_arc.angle = -math.tau * (1 - progress)
                        self._countdown_arc.opacity = int(127 * progress)
                        self._countdown_batch.draw()
                '''
            case FineState.GENERATING_IMAGE:
                self._frozen_pose_image.blit(0, 0, width=self.width, height=self.height)

                self._wait_batch.draw()
                for index, circle in enumerate(self._wait_circles):
                    circle.opacity = int(127 * (1 + math.sin(time.time() * math.pi - math.pi * index / 4)) / 2)

                if time_passed <= 1:
                    self._flash_rect.opacity = int(255 * (1 - time_passed))
                    self._flash_batch.draw()

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)

        if symbol == pyglet.window.key.SPACE:
            self.invokeGeneration()

    def getPoseImage(self):
        return pyglet.image.ImageData(
            self._pose_detector.frame_width,
            self._pose_detector.frame_height,
            "RGB",
            self._pose_detector.getFlippedPoseImageBytes(),
        )

    def invokeGeneration(self):
        self._server.invokeGeneration("starscream.local:9090", "resources/workflow.json")

        self._frozen_pose_image = self._pose_image
        self._frozen_pose_png_data = self._pose_detector.getPoseImagePNG()

        self.setState(FineState.GENERATING_IMAGE)

    def getFrozenPoseImagePNG(self):
        return self._frozen_pose_png_data

    def setInvokedImage(self, data):
        image = self._pose_detector.makeImage(data)

        (height, width) = image.shape[:2]
        with self._generated_image_lock:
            self._generated_image = pyglet.image.ImageData(width, height, "RGB", bytes(image))

        self.setState(FineState.SHOWING_RESULT)


if __name__ == "__main__":
    window = ThisIsFineWindow()
    try:
        pyglet.app.run()
    except (KeyboardInterrupt, Exception) as e:
        import traceback

        traceback.print_exception(e)
        window.cleanup()