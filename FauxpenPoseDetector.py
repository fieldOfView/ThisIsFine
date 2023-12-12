# Copyright 2023 Aldo Hoeben / fieldOfView
#
# roughly based on MediaPipe example code
#
# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time

from typing import Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


class FauxpenPoseDetector:
    def __init__(
        self,
        model: str,
        num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float,
        min_tracking_confidence: float,
        callback: Optional[callable] = None,
    ) -> None:
        """Run inference on images on demand.

        Args:
            model: Name of the pose landmarker model bundle.
            num_poses: Max number of poses that can be detected by the landmarker.
            min_pose_detection_confidence: The minimum confidence score for pose
              detection to be considered successful.
            min_pose_presence_confidence: The minimum confidence score of pose
              presence score in the pose landmark detection.
            min_tracking_confidence: The minimum confidence score for the pose
              tracking to be considered successful.
            callback: A python function to call when Mediapipe is done handling
              a queued image
        """

        # Initialize the pose landmarker model
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self.handle_result,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self._detected_landmarks: Optional[vision.PoseLandmarkerResult] = None
        self._detected_image: Optional[mp.Image] = None

        self._callback = callback

        self.radius = 3
        self.antialias = False

        self._fps_calculate_frames = 10
        self._fps_counter = 0
        self._fps = -1
        self._fps_start_time = time.time()

    def close(self) -> None:
        """
        Closes the detector.

        This method closes the detector and releases any resources used by it.
        """
        self._detector.close()

    def set_draw_options(self, radius: int = 3, antialias: bool = False) -> None:
            """
            Set the draw options for landmarks and connections.

            Args:
                radius (int): The radius to use for landmarks and connections when drawing.
                antialias (bool): Flag to enable antialiasing when drawing.
            
            Returns:
                None
            """
            self.radius = radius
            self.antialias = antialias

    def queue_image(self, rgb_image: np.ndarray) -> None:
            """
            Queues an RGB image for pose detection.

            Args:
                rgb_image (np.ndarray): The RGB image to be processed.

            Returns:
                None
            """
            # Run pose landmarker using the model.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            t = time.time_ns() // 1_000_000
            self._detector.detect_async(mp_image, t)

    def handle_result(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        """
        Handles the result of pose landmark detection.

        Args:
            result (vision.PoseLandmarkerResult): The result of pose landmark detection.
            output_image (mp.Image): The output image.
            timestamp_ms (int): The timestamp in milliseconds at the time the image was queued.

        Returns:
            None
        """
        # Calculate the FPS
        if self._fps_counter % self._fps_calculate_frames == 0:
            self._fps = self._fps_calculate_frames / (time.time() - self._fps_start_time)
            self._fps_start_time = time.time()
        self._fps_counter += 1

        # Store the landmarks and the image they were detected on
        self._detected_landmarks = result
        self._detected_image = output_image

    def get_results(self) -> Tuple[vision.PoseLandmarkerResult, np.ndarray]:
        """
        Returns the latest detected landmarks and the detected image (if available).

        Returns:
            A tuple containing the detected landmarks and the detected image (if available).
        """
        return (
            self._detected_landmarks,
            None if self._detected_image is None else self._detected_image.numpy_view()
        )

    def draw_poses_on_image(
            self,
            image: np.ndarray,
            poses: vision.PoseLandmarkerResult
        ) -> None:
            """Draws OpenPose-like landmarks and the connections on the image for all
            detected poses.

            Args:
                image: A three channel BGR image represented as numpy ndarray.
                poses: The poselandmarker result from the detector.

            Returns:
                None
            """
            for pose_landmarks in poses.pose_landmarks:
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                        for landmark in pose_landmarks
                    ]
                )
                self._draw_landmarks_on_image(image, pose_landmarks_proto)

    def _draw_landmarks_on_image(
        self,
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
    ) -> None:
        """Draws OpenPose-like landmarks and the connections on the image.

        Args:
            image: A three channel BGR image represented as numpy ndarray.
            landmark_list: A normalized landmark list proto message to be annotated on the image.

        Raises:
            ValueError: If one of the followings:
                a) If the input image is not three channel BGR.
                b) If any connetions contain invalid landmark index.
        """

        # For each marker in the OpenPose model, the index of the corresponding MediaPipe marker and the BGR color
        _OPENPOSE_MARKERS = [
            (0, [0, 0, 255]),  # nose
            (33, [0, 85, 255]),  # neck (center between 11 and 12)
            (12, [0, 170, 255]),  # right shoulder
            (14, [0, 255, 255]),  # right elbow
            (16, [0, 255, 170]),  # right wrist
            (11, [0, 255, 85]),  # left shoulder
            (13, [0, 255, 0]),  # left elbow
            (15, [85, 255, 0]),  # left wrist
            (24, [170, 255, 0]),  # right hip
            (26, [255, 255, 0]),  # right knee
            (28, [255, 170, 0]),  # right ankle
            (23, [255, 85, 0]),  # left hip
            (25, [255, 0, 0]),  # left knee
            (27, [255, 0, 85]),  # left ankle
            (5, [255, 0, 170]),  # right eye
            (2, [255, 0, 255]),  # left eye
            (8, [170, 0, 255]),  # right ear
            (7, [85, 0, 255]),  # left ear
        ]

        # For each connection in the OpenPose model, the OpenPose indices and the BGR color
        _OPENPOSE_CONNECTIONS = [
            (1, 2, [0, 0, 153]),  # right shoulderblade
            (1, 5, [0, 51, 153]),  # left shoulderblade
            (2, 3, [0, 102, 153]),  # right arm
            (3, 4, [0, 153, 153]),  # right forearm
            (5, 6, [0, 125, 102]),  # left arm
            (6, 7, [0, 153, 51]),  # left forearm
            (1, 8, [0, 153, 0]),  # right torso
            (8, 9, [51, 153, 0]),  # right upper leg
            (9, 10, [102, 153, 0]),  # right lower leg
            (1, 11, [153, 153, 0]),  # left torso
            (11, 12, [153, 102, 0]),  # left upper leg
            (12, 13, [153, 51, 0]),  # left lower leg
            (1, 0, [153, 0, 0]),  # head
            (0, 14, [153, 0, 51]),  # right eyebrow
            (14, 16, [153, 0, 102]),  # right ear
            (0, 15, [153, 0, 153]),  # left eyebrow
            (15, 17, [102, 0, 153]),  # left ear
        ]

        if not landmark_list:
            return

        if image.shape[2] != 3:
            raise ValueError("Input image must contain three channel bgr data.")
        image_rows, image_cols, _ = image.shape

        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

        # Synthesize the neck landmark from the left and right shoulder.
        # Mediapipe does not have a landmark for the neck.
        try:
            idx_to_coordinates[33] = (
                int((idx_to_coordinates[11][0] + idx_to_coordinates[12][0]) / 2),
                int((idx_to_coordinates[11][1] + idx_to_coordinates[12][1]) / 2),
            )
        except KeyError:
            # either the left or right shoulder are not in view
            pass

        num_landmarks = len(landmark_list.landmark) + 1  # one extra for the neck synthetic landmark
        # Draws the connections if the start and end landmarks are both visible.
        for connection in _OPENPOSE_CONNECTIONS:
            start_idx = _OPENPOSE_MARKERS[connection[0]][0]
            end_idx = _OPENPOSE_MARKERS[connection[1]][0]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )

            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                start_coordinates = idx_to_coordinates[start_idx]
                end_coordinates = idx_to_coordinates[end_idx]
                connection_center = (
                    int((start_coordinates[0] + end_coordinates[0]) / 2),
                    int((start_coordinates[1] + end_coordinates[1]) / 2),
                )
                connection_radius = int(
                    math.sqrt(
                        (start_coordinates[0] - end_coordinates[0]) ** 2
                        + (start_coordinates[1] - end_coordinates[1]) ** 2
                    )
                    / 2
                )
                connection_angle = math.degrees(
                    math.atan2(start_coordinates[1] - end_coordinates[1], start_coordinates[0] - end_coordinates[0])
                )
                cv2.ellipse(
                    image,
                    connection_center,
                    (connection_radius, self.radius),
                    connection_angle,
                    0, 360,  # section
                    connection[2],  # color
                    -1,  # thickness -> filled
                    cv2.LINE_AA if self.antialias else cv2.LINE_8,
                )

        # Draws landmark points after finishing the connection lines
        for marker in _OPENPOSE_MARKERS:
            if marker[0] not in idx_to_coordinates:
                continue
            if not self.antialias:
                cv2.circle(image, idx_to_coordinates[marker[0]], self.radius, marker[1], -1)
            else:
                # cv2.circle does not have an option to draw antialiased, but cv2.ellipse does
                cv2.ellipse(
                    image,
                    idx_to_coordinates[marker[0]],
                    (self.radius, self.radius),
                    0,  # angle
                    0, 360,  # section
                    marker[1],  # color
                    -1,  # thickness -> filled
                    cv2.LINE_AA,
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model", help="Name of the pose landmarker model bundle.", required=False, default="resources/pose_landmarker.task"
    )
    parser.add_argument(
        "--numPoses", help="Max number of poses that can be detected by the landmarker.", required=False, default=1
    )
    parser.add_argument(
        "--minPoseDetectionConfidence",
        help="The minimum confidence score for pose detection to be considered " "successful.",
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--minPosePresenceConfidence",
        help="The minimum confidence score of pose presence score in the pose " "landmark detection.",
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--minTrackingConfidence",
        help="The minimum confidence score for the pose tracking to be " "considered successful.",
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--fullscreen", help="Show result fullscreen.", action="store_true",
    )
    parser.add_argument(
        "--hideVideo", help="Show tracked landmarks on a black background.", action="store_true",
    )

    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument("--cameraId", help="Id of camera.", required=False, default=0)
    parser.add_argument("--frameWidth", help="Width of frame to capture from camera.", required=False, default=0)
    parser.add_argument("--frameHeight", help="Height of frame to capture from camera.", required=False, default=0)
    parser.add_argument("--frameDelay", help="The time in ms to wait between captures", required=False, default=20)
    args = parser.parse_args()

    detector = FauxpenPoseDetector(
        model=args.model,
        num_poses=args.numPoses,
        min_pose_detection_confidence=args.minPoseDetectionConfidence,
        min_pose_presence_confidence=args.minPosePresenceConfidence,
        min_tracking_confidence=args.minTrackingConfidence,
    )

    source = args.cameraId
    #source = "testing/dance1.mp4"

    if args.fullscreen:
        cv2.namedWindow("pose", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(source)
    if args.frameWidth != 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.frameWidth))
    if args.frameHeight != 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.frameHeight))

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector.queue_image(rgb_image)

        (poses, processed_image) = detector.get_results()
        if poses is not None:
            if args.hideVideo:
                show_image = np.zeros_like(rgb_image)
            else:
                show_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            detector.draw_poses_on_image(show_image, poses)
            cv2.imshow("pose", show_image)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(int(args.frameDelay)) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()