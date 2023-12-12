from FauxpenPoseDetector import FauxpenPoseDetector

import threading
import cv2
import numpy as np


class FineDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        self.stopRequest = threading.Event()

        self.detector = FauxpenPoseDetector(
            model="resources/pose_landmarker.task",
            num_poses=3,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.source = 0#"testing/dance1.mp4"
        self.frame_width = 640
        self.frame_height = 360
        self.frame_delay = 20

        self.loop = True
        self.debug = False

        self.lock = threading.Lock()
        self.pose_image = None
        self.full_pose_detected = False
        self.start()

    def requestStop(self):
        self.stopRequest.set()

    def getFullPoseDetected(self):
        with self.lock:
            return self.full_pose_detected

    def getFlippedPoseImageBytes(self):
        with self.lock:
            if self.pose_image is not None:
                return bytes(cv2.cvtColor(cv2.flip(self.pose_image, 0), cv2.COLOR_BGR2RGB))
            else:
                return None

    def getPoseImagePNG(self):
        with self.lock:
            if self.pose_image is not None:
                return bytes(cv2.imencode(".png", self.pose_image)[1])
            else:
                return None

    def makeImage(self, data):
        # convert image data buffer to uint8
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)

        return image

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if self.frame_width != 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height != 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Continuously capture images from the camera and run inference
        while cap.isOpened() and not self.stopRequest.is_set():
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

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.detector.queueImage(rgb_image)

            (poses, processed_image) = self.detector.getResults()
            pose_image = np.zeros_like(rgb_image)
            with self.lock:
                self.full_pose_detected = False
                if poses is not None:
                    self.detector.drawPosesOnImage(pose_image, poses)
                    if self.debug:
                        cv2.imshow("pose", pose_image)

                    self.pose_image = pose_image
                    self.full_pose_detected = any([len(p) >= 33 for p in poses.pose_landmarks])

            cv2.waitKey(int(self.frame_delay))
