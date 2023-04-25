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

    def __init__(self, original_frame, landmarks, side):
        self.frame = None
        # self.brow_outside = None
        # self.brow_inside = None
        self.position = None
        self.curve = None # options will be: upturn, downturn, neutral

        self.landmark_points = None

        self._analyze(original_frame, landmarks, side)

    def _position(self, landmarks, points, side):
        ### Take the center of the eyebrow to indicate the overall position.
        #   Then determine the curve based on the two ends.

        if side == 0: # left
            outter_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
            inner_point = (landmarks.part(points[-1]).x, landmarks.part(points[-1]).y)
        elif side == 1: # right
            outter_point = (landmarks.part(points[-1]).x, landmarks.part(points[-1]).y)
            inner_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)

        middle_point = (landmarks.part(points[2]).x, landmarks.part(points[2]).y)

        # Remember, images are indexed such that smaller y values are nearer to the top.
        # ...I think.
        threshold = 2
        if inner_point[1] - outter_point[1] >= 2:
            self.curve = 'down'
        elif outter_point[1] - inner_point[1] >= 2:
            self.curve = 'up'
        else:
            self.curve = 'na'

        return middle_point

    def _analyze(self, original_frame, landmarks, side):
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
        elif side == 1:
            points = self.RIGHT_BROW_POINTS
        else:
            return

        self.position = self._position(landmarks, points, side)

