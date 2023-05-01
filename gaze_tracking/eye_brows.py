import math
import numpy as np
import cv2
from .pupil import Pupil


class EyeBrow(object):
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
        self.distance_from_nose = None
        self.position = None

        self.landmarks = landmarks
        self.side = side
        self.vect_calc = vect_calc
        self.dimension_calc = dimension_calc

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

    def annotated_frame(self, frame):
        """Returns the main frame with pupils highlighted"""
        # frame = self.frame.copy()

        if self.position is not None:
            color = (255, 0, 0)
            point = self.position
            cv2.line(frame, (point[0] - 10, point[1]), (point[0] + 10, point[1]), color)

        return frame

    def draw_vect(self, frame):
        if self.position:
            point = self.position
            start, end, dist = self.vect_calc.find_vector(point)

            point_2d = self.dimension_calc.to_2d([start, end])
            # point_2d = np.int32(point_2d.reshape(-1, 2))
            frame = cv2.line(
                frame,
                tuple(point_2d[0]),
                tuple(point_2d[1]),
                (0, 255, 0), 2, cv2.LINE_AA
            )

        return frame
