from typing import Optional

import threading
import cv2
import numpy as np

from fauxpenposedetector import FauxpenPoseDetector

class FineDetector(threading.Thread):
    def __init__(self) -> None:
        threading.Thread.__init__(self)

        self.stop_request = threading.Event()

        self.detector = FauxpenPoseDetector(
            model="resources/pose_landmarker.task",
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector.set_draw_options(antialias=True, radius=2)

        self.source = 0 #"testing/dance1.mp4"
        self.frame_width = 640
        self.frame_height = 360
        self.frame_delay = 10

        self.loop = True
        self.debug = False

        self.lock = threading.Lock()
        self.pose_image = None
        self.canny_image = None
        self.full_pose_detected = False
        self.start()

    def request_stop(self) -> None:
        """
        Requests the FineDetector to stop processing, ending its threaded loop.
        """
        self.stop_request.set()

    def get_full_pose_detected(self) -> bool:
        """
        Returns a boolean indicating whether the full pose is detected.

        Returns:
            bool: True if the full pose is detected, False otherwise.
        """
        with self.lock:
            return self.full_pose_detected

    def get_flipped_pose_image_bytes(self) -> Optional[bytes]:
        """
        Returns the flipped pose image as bytes.

        Returns:
            Optional[bytes]: The flipped pose image as bytes, or None if the pose image is not available.
        """
        with self.lock:
            if self.pose_image is not None:
                return bytes(cv2.cvtColor(cv2.flip(self.pose_image, 0), cv2.COLOR_BGR2RGB))
            else:
                return None

    def get_pose_image_png(self) -> Optional[bytes]:
        """
        Returns the pose image in PNG format as bytes.

        Returns:
            Optional[bytes]: The pose image in PNG format as bytes, or None if the pose image is not available.
        """
        with self.lock:
            if self.pose_image is not None:
                return bytes(cv2.imencode(".png", self.pose_image)[1])
            else:
                return None

    def get_canny_image_png(self) -> Optional[bytes]:
        """
        Returns the Canny image in PNG format.

        Returns:
            bytes: The Canny image in PNG format, or None if the Canny image is not available.
        """
        with self.lock:
            if self.canny_image is not None:
                return bytes(cv2.imencode(".png", self.canny_image)[1])
            else:
                return None

    def make_image(self, data: bytes) -> np.ndarray:
            """
            Converts image data buffer to a numpy array.

            Args:
                data (bytes): The image data buffer.

            Returns:
                np.ndarray: The converted image as a numpy array.
            """
            # convert image data buffer to uint8
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)

            return image

    def run(self):
        """
        Runs the FineDetector by capturing images from the camera and performing inference.

        This method continuously captures images from the camera and runs inference using the FineDetector.
        It sets the frame width and height if provided, and then processes each captured image.
        The processed image is used to detect poses and draw them on the image if any poses are detected.
        The pose image and canny image are updated accordingly.

        Returns:
            None
        """
        cap = cv2.VideoCapture(self.source)
        if self.frame_width != 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height != 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Continuously capture images from the camera and run inference
        while cap.isOpened() and not self.stop_request.is_set():
            success, image = cap.read()
            if not success:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            image_shape = image.shape[:2]
            if image_shape != (self.frame_width, self.frame_height):
                if self.frame_width == 0 or self.frame_height == 0:
                    self.frame_height, self.frame_width = image_shape
                else:
                    image = cv2.resize(image, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.detector.queue_image(rgb_image)

            (poses, processed_image) = self.detector.get_results()
            pose_image = np.zeros_like(rgb_image)
            with self.lock:
                # Rest of the code...
                self.full_pose_detected = False
                if poses is not None:
                    self.detector.draw_poses_on_image(pose_image, poses)
                    if self.debug:
                        cv2.imshow("pose", pose_image)

                    self.pose_image = pose_image
                    self.canny_image = cv2.Canny(rgb_image, 100, 200)
                    self.full_pose_detected = any([len(p) >= 33 for p in poses.pose_landmarks])

            cv2.waitKey(int(self.frame_delay))
