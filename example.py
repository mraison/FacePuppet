"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import cv2
import os
import dlib
from gaze_tracking.face import Face
import numpy as np

face_detector = dlib.get_frontal_face_detector()  # mraison here's where the magic happens
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "training_data_NEW/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)

webcam = cv2.VideoCapture(0)
frame_size = (
    webcam.get(cv2.CAP_PROP_FRAME_WIDTH),
    webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
prev_landmarks=None
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

    use_prev = True
    if landmarks and prev_landmarks:
        neighborhood = 10
        for i in Face.FEATURE_IDS_FOR_HEAD_TILT:
            hypot = (
                (landmarks.part(i).x - prev_landmarks.part(i).x) ** 2
                + (landmarks.part(i).y - prev_landmarks.part(i).y) ** 2
            ) ** (1 / 2)
            if abs(hypot) - neighborhood > 0:
                use_prev = False

        if use_prev:
            landmarks = prev_landmarks
            for i in range(67):
                prev_landmarks.part(i).x = (landmarks.part(i).x + prev_landmarks.part(i).x)/2
                prev_landmarks.part(i).y = (landmarks.part(i).y + prev_landmarks.part(i).y)/2

    face = Face(frame, frame_size, landmarks)
    face.analyze()
    frame = face.annotate(frame)
    frame = face.draw_vecs(frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

    cv2.imshow("Demo", frame)
    if not use_prev:
        prev_landmarks = landmarks

webcam.release()
cv2.destroyAllWindows()
