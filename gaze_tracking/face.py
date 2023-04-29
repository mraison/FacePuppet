import numpy as np
import cv2
from .eye import Eye
from .calibration import Calibration
from .eye_brows import EyeBrow
from .mouth import Mouth
from imutils import face_utils


###############################
# ripped from
# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# and
# https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
###############################

class CameraSpecs(object):
    def __init__(self, dimensions):
        focal_length = dimensions[1]
        center = (dimensions[1] / 2, dimensions[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Not sure what these are but they were values commonly floating around.
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]]
        )

class Face(object):
    FEATURE_IDS_FOR_HEAD_TILT = [
        33,  # Nose tip
        8,  # Chin
        36,  # Left eye left corner
        45,  # Right eye right corner
        48,  # Left Mouth corner
        54  # Right mouth corner
    ]

    # 3D model points.
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion

    def __init__(self, frame, dimensions, landmarks):
        self.frame = frame
        self.greyframe = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self.calibration = Calibration()
        # Camera internals
        self.camera_specs = CameraSpecs(dimensions)

        self.brows = (
            EyeBrow(landmarks, 0),
            EyeBrow(landmarks, 1)
        )
        self.eyes = (
            Eye(self.greyframe, landmarks, 0, self.calibration),
            Eye(self.greyframe, landmarks, 1, self.calibration)
        )
        self.mouth = Mouth(landmarks)

        self.landmarks = landmarks

        self.face_shape = None
        # These will inform us how to adjust the analysis of all other parts of the face.
        self.rotational_vector = None
        self.translation_vector = None

    def is_detected(self):
        if self.rotational_vector is None or self.translation_vector is None:
            return False

        return True

    def analyze(self):
        if self.landmarks is None:
            self.rotational_vector = None
            self.translation_vector = None
            return

        self._analyze()
        self.brows[0].analyze()
        self.brows[1].analyze()
        self.eyes[0].analyze()
        self.eyes[1].analyze()
        self.mouth.analyze()

    def _analyze(self):
        # grab the face shape...sort of unrelated to the rest of this function...
        self.face_shape = face_utils.shape_to_np(self.landmarks)

        image_points = np.array(
            [
                (self.landmarks.part(pt).x, self.landmarks.part(pt).y) for pt in self.FEATURE_IDS_FOR_HEAD_TILT
            ], dtype=np.float32
        )

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.MODEL_POINTS, image_points, self.camera_specs.camera_matrix,
            self.DIST_COEFFS, rvec=self.camera_specs.r_vec, tvec=self.camera_specs.t_vec,
            flags=cv2.SOLVEPNP_ITERATIVE,
            # useExtrinsicGuess = True
        )
        self.rotational_vector = rotation_vector
        self.translation_vector = translation_vector

    def draw_annotation_box(
            self,
            image,
            color=(0, 255, 0),
            line_width=2
    ):
        if not self.is_detected():
            return image
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          self.rotational_vector,
                                          self.translation_vector,
                                          self.camera_specs.camera_matrix,
                                          self.DIST_COEFFS)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        return image

    def annotated_face_shape(self, frame):
        """Returns the main frame with pupils highlighted"""
        # frame = self.frame.copy()
        if self.face_shape is not None and self.face_shape.any():
            for (x, y) in self.face_shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        return frame

    def annotate(self, frame):
        try:
            frame = self.draw_annotation_box(frame)
            frame = self.annotated_face_shape(frame)
            frame = self.brows[0].annotated_frame(frame)
            frame = self.brows[1].annotated_frame(frame)
            frame = self.eyes[0].annotated_frame(frame)
            frame = self.eyes[1].annotated_frame(frame)
            frame = self.mouth.annotated_frame(frame)
            return frame
        except:
            return frame
