import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration, vect_calc, dimension_calc):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self.original_frame = original_frame
        self.landmarks = landmarks
        self.side = side
        self.calibration = calibration
        self.vect_calc = vect_calc
        self.dimension_calc = dimension_calc

    def is_located(self):
        return self.pupils_located
    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.pupil.x)
            int(self.pupil.y)
            return True
        except Exception:
            return False

    def pupil_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.origin[0] + self.pupil.x
            y = self.origin[1] + self.pupil.y
            return (x, y)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            return self.blinking > 3.8

    def analyze(self):
        if self.landmarks is None:
            self.frame = None
            self.origin = None
            self.center = None
            self.pupil = None
            return
        self._analyze(self.original_frame, self.landmarks, self.side, self.calibration)

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

    def annotated_frame(self, frame):
        """Returns the main frame with pupils highlighted"""
        if self.pupils_located:
            color = (0, 255, 0)
            x, y = self.pupil_coords()
            cv2.line(frame, (x - 5, y), (x + 5, y), color)
            cv2.line(frame, (x, y - 5), (x, y + 5), color)

        return frame

    def draw_vect(self, frame):
        if self.pupils_located:
            point = self.pupil_coords()
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

