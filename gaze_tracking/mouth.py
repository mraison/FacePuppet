import math
import numpy as np
import cv2


class Mouth(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    TOP_LIP_OUTER_POINTS = [48, 49, 50, 51, 52, 53, 54]
    TOP_LIP_INNER_POINTS = [60, 61, 62, 63, 64]
    BOTTOM_LIP_INNER_POINTS = [60, 67, 66, 65, 64]
    BOTTOM_LIP_OUTER_POINTS = [48, 59, 58, 57, 56, 55, 54]

    def __init__(self, original_frame, landmarks):
        self.frame = None
        self.mouth_shape = None
        self.landmark_points = None
        self.mouth_center_location = None

        self._analyze(original_frame, landmarks)

    def is_closed(self):
        threshold = 4
        return self.mouth_shape and 0 < self.mouth_shape[1] <= threshold

    def is_half_open(self):
        threshold = 12
        return self.mouth_shape and 4 < self.mouth_shape[1] <= threshold

    def is_full_open(self):
        threshold = 20
        return self.mouth_shape and 12 < self.mouth_shape[1] <= threshold

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _mouth_shape(self, landmarks, upper_lip, lower_lip):
        ## take the center 3 points of the upper and lower lips
        # assume ordered left to write.
        x_top_l = [landmarks.part(i).x for i in upper_lip]
        y_top_l = [landmarks.part(i).y for i in upper_lip]
        x_bottom_l = [landmarks.part(i).x for i in lower_lip]
        y_bottom_l = [landmarks.part(i).y for i in lower_lip]
        av_top_lip = (np.mean(x_top_l), np.mean(y_top_l))
        av_bottom_lip = (np.mean(x_bottom_l), np.mean(y_bottom_l))

        lip_gap_height = av_bottom_lip[1] - av_top_lip[1]
        lip_gap_width = np.mean([(x_top_l[-1] - x_top_l[0]), (x_bottom_l[-1] - x_bottom_l[0])])

        self.mouth_center_location = self._middle_point(av_top_lip, av_bottom_lip)

        return (lip_gap_width, lip_gap_height)

    def _analyze(self, original_frame, landmarks):
        self.mouth_shape = self._mouth_shape(
            landmarks,
            self.TOP_LIP_INNER_POINTS,
            self.BOTTOM_LIP_INNER_POINTS
        )
