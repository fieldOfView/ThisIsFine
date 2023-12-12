# This is Fine!
### an interactive InvokeAI frontend

This is the code for Aldo Hoeben's "This is Fine!" installation, first shown in the "Elementen" exhibition at Stichting KunstWerkt in Schiedam (NL).

The installation shows visitors of the gallery engulfed in flames in images created with [InvokeAI](https://invoke.ai/). It is a reference to the [webmeme by the same name](https://knowyourmeme.com/memes/this-is-fine), which is based on a webcomic by [K. C. Green](https://knowyourmeme.com/memes/people/kc-green).

Technically the installation shows images created with InvokeAI, based on the poses and a canny representation of a camera view. Pose estimations are shown in realtime as an overlay over the previously generated image.

## Installation

The application can be run on the same server that runs the InvokeAI instance, or on another server that can be accessed by the server that runs InvokeAI (and vice-versa). It is highly recommended that the application is run from its own virtual environment.

Windows:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python fine.py
```

Linux/MacOS:
```
python3 -m venv venv
source venv\bin\activate.sh
pip install -r requirements.txt
python3 fine.py
```

The frontend sends a workflow json file from the `resources` folder to the InvokeAI instance.

## Classes and other elements of note:

### FauxpenPoseDetector

FauxpenPoseDetector is a realtime pose detector based on the [mediapipe pose estimator](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), which is transmogrified into an OpenPose-alike result which can be used by the [OpenPose Controlnet](https://huggingface.co/lllyasviel/sd-controlnet-openpose).

The FauxpenPoseDetector requires the `pose_landmarker.task` file placed in the `resources` folder. It can be downloaded from [developers.google.com](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) (with your choice of a lite, full or heavy model).

The FauxpenPoseDetector can be tested standalone in the commandline:

```
usage: fauxpenposedetector.py [-h] [--model MODEL] [--numPoses NUMPOSES] [--minPoseDetectionConfidence MINPOSEDETECTIONCONFIDENCE] [--minPosePresenceConfidence MINPOSEPRESENCECONFIDENCE]
                              [--minTrackingConfidence MINTRACKINGCONFIDENCE] [--fullscreen] [--hideVideo] [--cameraId CAMERAID] [--frameWidth FRAMEWIDTH] [--frameHeight FRAMEHEIGHT]
                              [--frameDelay FRAMEDELAY]

options:
  -h, --help            show this help message and exit
  --model MODEL         Name of the pose landmarker model bundle. (default: resources/pose_landmarker.task)
  --numPoses NUMPOSES   Max number of poses that can be detected by the landmarker. (default: 1)
  --minPoseDetectionConfidence MINPOSEDETECTIONCONFIDENCE
                        The minimum confidence score for pose detection to be considered successful. (default: 0.5)
  --minPosePresenceConfidence MINPOSEPRESENCECONFIDENCE
                        The minimum confidence score of pose presence score in the pose landmark detection. (default: 0.5)
  --minTrackingConfidence MINTRACKINGCONFIDENCE
                        The minimum confidence score for the pose tracking to be considered successful. (default: 0.5)
  --fullscreen          Show result fullscreen. (default: False)
  --hideVideo           Show tracked landmarks on a black background. (default: False)
  --cameraId CAMERAID   Id of camera. (default: 0)
  --frameWidth FRAMEWIDTH
                        Width of frame to capture from camera. (default: 0)
  --frameHeight FRAMEHEIGHT
                        Height of frame to capture from camera. (default: 0)
  --frameDelay FRAMEDELAY
                        The time in ms to wait between captures (default: 20)
```

### FineDetector

A wrapper that runs the FauxpenPoseDetector in a thread and provides results of the detector to the frontend.

### FineServer

A simple Flask server which serves poses and canny images to InvokeAI, and which handles newly created images. It also contains a small utility method to convert workflow JSON files to a batch that gets enqueued to InvokeAI.

### Remote Image node

The application relies on communication via a simple REST interface between the This is Fine! frontend and the InvokeAI backend. It uses the [Remote Image nodes](https://github.com/fieldOfView/InvokeAI-remote_image). The provided workflow in the `resources` folder uses these nodes.

### warpcalibration

To facilitate running the installation as a projection on a large wall, a quad warp / corner pinning / keystone utility is provided. It can be calibrated using a small application
```
python warpcalibration.py
```
Select a corner with the keys `1`, `2`, `3` or `4` and use the cursor keys to move that corner.

Press `s` to save the calibration, `r` to reset to the full projection rectangle and `l` to restore a previousl saved calibration.

The This is Fine! application automatically loads a warpcalibration if it is present in the `resources` folder.

