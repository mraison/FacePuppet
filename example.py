"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import os
import dlib
from gaze_tracking.gaze_tracking import GazeTracking
from gaze_tracking.mouth_tracking import MouthTracking

face_detector = dlib.get_frontal_face_detector()  # mraison here's where the magic happens
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "training_data_NEW/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)

gaze = GazeTracking()
mouth_track = MouthTracking()
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    try:
        faces = face_detector(frame)
        landmarks = predictor(frame, faces[0])
    except:
        landmarks = None

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame, landmarks)
    frame = gaze.annotated_frame(frame)

    mouth_track.refresh(frame, landmarks)
    frame = mouth_track.annotated_frame(frame)
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
