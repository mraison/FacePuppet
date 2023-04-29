"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import os
import dlib
from gaze_tracking.gaze_tracking import GazeTracking
from gaze_tracking.mouth_tracking import MouthTracking
from gaze_tracking.brow_tracker import BrowTracking
from gaze_tracking.face import Face
import numpy as np

face_detector = dlib.get_frontal_face_detector()  # mraison here's where the magic happens
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "training_data_NEW/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)

gaze = GazeTracking()
mouth_track = MouthTracking()
brow_tracker = BrowTracking()
webcam = cv2.VideoCapture(0)

frame_size = (
    webcam.get(cv2.CAP_PROP_FRAME_WIDTH),
    webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
# face_model_points = get_full_model_points(
#     os.path.abspath(os.path.join(cwd, "training_data_NEW/model.txt"))
# )

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # random drop
    if np.random.randint(10) in [0, 1, 3]:
        continue

    try:
        faces = face_detector(frame)
        landmarks = predictor(frame, faces[0])
    except:
        landmarks = None

    face = Face(frame, frame_size, landmarks)
    face.analyze()
    frame = face.annotate(frame)

    # We send this frame to GazeTracking to analyze it
    # gaze.refresh(frame, landmarks)
    # frame = gaze.annotated_frame(frame)
    #
    # mouth_track.refresh(frame, landmarks)
    # frame = mouth_track.annotated_frame(frame)
    #
    # brow_tracker.refresh(frame, landmarks)
    # frame = brow_tracker.annotated_frame(frame)
    #
    # gaze_text = ""
    # if gaze.is_blinking():
    #     gaze_text = "Blinking"
    # elif gaze.is_right():
    #     gaze_text = "Looking right"
    # elif gaze.is_left():
    #     gaze_text = "Looking left"
    # elif gaze.is_center():
    #     gaze_text = "Looking center"
    #
    # mouth_text = ""
    # if mouth_track.is_closed():
    #     mouth_text = "mouth closed"
    # elif mouth_track.is_half_open():
    #     mouth_text = "mouth half open"
    # elif mouth_track.is_full_open():
    #     mouth_text = "mouth full open"
    #
    # brow_text = ""
    # if brow_tracker.are_raised():
    #     brow_text = "brow raised"
    # elif brow_tracker.are_neutral():
    #     brow_text = "brow neutral"
    # elif brow_tracker.are_furrowed():
    #     brow_text = "brow furrowed"
    #
    # cv2.putText(frame, gaze_text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    # cv2.putText(frame, mouth_text, (90, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    # cv2.putText(frame, brow_text, (90, 180), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # left_pupil = gaze.pupil_left_coords()
    # right_pupil = gaze.pupil_right_coords()
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        print(
            brow_tracker.right_brow.distance_from_nose
        )
        print(
            brow_tracker.left_brow.distance_from_nose
        )
        break
   
webcam.release()
cv2.destroyAllWindows()
