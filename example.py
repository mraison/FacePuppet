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
    # frame = face.annotate(frame)
    frame = face.draw_vecs(frame)

    k = cv2.waitKey(1)
    # if k == ord('p'):
    #     pt = np.array([
    #         [face.eyes[0].origin[0] + face.eyes[0].pupil.x],
    #         [face.eyes[0].origin[1] + face.eyes[0].pupil.y],
    #         [1]
    #     ], dtype="double")
    #     print(pt)
    #     print(face.camera_specs.dimensions)
    # try:
    #     uvPoint1 = np.array([
    #         [face.eyes[0].origin[0] + face.eyes[0].pupil.x],
    #         [face.eyes[0].origin[1] + face.eyes[0].pupil.y],
    #         [1]
    #     ], dtype="double")
    #     uvPoint2 = np.array([
    #         [face.eyes[1].origin[0] + face.eyes[1].pupil.x],
    #         [face.eyes[1].origin[1] + face.eyes[1].pupil.y],
    #         [1]
    #     ], dtype="double")
    #
    #     s, rot_mat, c_mat, t_vec = face.calculate_2d_to_3d_translation_matrix(
    #         face.rotational_vector,
    #         face.translation_vector
    #     )
    #     res1 = np.dot(
    #         rot_mat,
    #         np.dot(
    #             np.dot(
    #                 s,
    #                 c_mat
    #             ),
    #             uvPoint1
    #         ) - t_vec
    #     )
    #     res2 = np.dot(
    #         rot_mat,
    #         np.dot(
    #             s*c_mat,
    #             uvPoint2
    #         ) - t_vec
    #     )
    #     # print(res1)
    #     # print(res2)
    #     point_3d = [res1, res2]
    #     point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)
    #     (point_2d, _) = cv2.projectPoints(point_3d,
    #                                       face.rotational_vector,
    #                                       face.translation_vector,
    #                                       face.camera_specs.camera_matrix,
    #                                       face.DIST_COEFFS)
    #     point_2d = np.int32(point_2d.reshape(-1, 2))
    #
    #     print("2D points: %s" % point_2d)
    #     print("Frame size %s | %s" % (frame_size[0], frame_size[1]))
    #     # Draw all the lines
    #     frame = cv2.line(frame, tuple(point_2d[0]), tuple(
    #         point_2d[1]), (0, 255, 0), 2, cv2.LINE_AA)
    #     dist_part = np.sum((res1-res2)**2, axis=0)
    #     # print(dist_part)
    #     dist = np.sqrt(dist_part)
    #     # print(
    #     #     "rotations: %s | %s | %s | %s" % (s, res1, res2, dist)
    #     # )
    # except Exception as e:
    #     print("passing eye trace...")
    #     print(e)

    if k == 27:
        break

    cv2.imshow("Demo", frame)

webcam.release()
cv2.destroyAllWindows()
