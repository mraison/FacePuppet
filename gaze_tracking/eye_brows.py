import numpy as np
import cv2
from .base import BaseFaceFeature


class EyeBrow(BaseFaceFeature):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_BROW_POINTS = [17, 18, 19, 20, 21]
    RIGHT_BROW_POINTS = [22, 23, 24, 25, 26]

    LEFT_TOP_EYELID = [37, 38]
    RIGHT_TOP_EYELID = [44, 43]

    BRIDGE_OF_NOSE = [27]

    def __init__(self, landmarks, side, vect_calc, dimension_calc):
        super().__init__(vect_calc, dimension_calc)
        self._slope = -1 if side == 0 else 1
        self.x_coeff = 1 if side == 0 else -1

        self.distance_from_nose = None
        self.position = None

        self.landmarks = landmarks
        self.side = side

    def is_located(self):
        return self.position is not None

    def is_raised(self):
        if self.distance_from_nose:
            return self.distance_from_nose > 35
        return False

    def is_neutral(self):
        if self.distance_from_nose:
            return 28 < self.distance_from_nose <= 35
        return False

    def is_furrowed(self):
        if self.distance_from_nose:
            return self.distance_from_nose <= 28
        return False

    def _position(self, landmarks, brow_points, nose_bridge):
        middle_point = (landmarks.part(brow_points[2]).x, landmarks.part(brow_points[2]).y)

        self.distance_from_nose = np.abs(
            middle_point[1] - landmarks.part(nose_bridge[0]).y
        )

        return middle_point

    def analyze(self):
        if self.landmarks is None:
            self.distance_from_nose = None
            self.position = None
            return
        self._analyze(self.landmarks, self.side)

    def _analyze(self, landmarks, side):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_BROW_POINTS
            eyelid = self.LEFT_TOP_EYELID
        elif side == 1:
            points = self.RIGHT_BROW_POINTS
            eyelid = self.RIGHT_TOP_EYELID
        else:
            return

        self.position = self._position(landmarks, points, eyelid)
        self.set_feature_reference_point(self.position)

    def annotated_frame(self, frame):
        """Returns the main frame with pupils highlighted"""
        # frame = self.frame.copy()

        if self.position is not None:
            color = (255, 0, 0)
            point = self.position
            cv2.line(frame, (point[0] - 10, point[1]), (point[0] + 10, point[1]), color)

        return frame
