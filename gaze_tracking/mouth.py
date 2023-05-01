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

    def __init__(self, landmarks, vect_calc, dimension_calc):
        self.mouth_shape = None
        self.landmark_points = landmarks
        self.vect_calc = vect_calc
        self.dimension_calc = dimension_calc
        self.mouth_center_location = None

    def is_located(self):
        """Check that the pupils have been located"""
        try:
            return len(self.mouth_center_location) == 2 and self.mouth_shape is not None
        except Exception:
            return False

    def is_closed(self):
        if self.mouth_shape:
            return self.mouth_shape[1] / self.mouth_shape[0] <= 0.05
        else:
            return True

    def is_half_open(self):
        if self.mouth_shape:
            return 0.05 < self.mouth_shape[1] / self.mouth_shape[0] < 0.2
        else:
            return False

    def is_full_open(self):
        if self.mouth_shape:
            return self.mouth_shape[1] / self.mouth_shape[0] >= 0.2
        else:
            return False

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

        lip_gap_height = np.absolute(av_top_lip[1] - av_bottom_lip[1])
        lip_gap_width = np.mean([np.absolute(x_top_l[-1] - x_top_l[0]), np.absolute(x_bottom_l[-1] - x_bottom_l[0])])

        self.mouth_center_location = self._middle_point(av_top_lip, av_bottom_lip)

        return (lip_gap_width, lip_gap_height)

    def analyze(self):
        if self.landmark_points is None:
            self.mouth_shape = None
            self.mouth_center_location = None
            return
        self._analyze(self.landmark_points)

    def _analyze(self, landmarks):
        self.mouth_shape = self._mouth_shape(
            landmarks,
            self.TOP_LIP_INNER_POINTS,
            self.BOTTOM_LIP_INNER_POINTS
        )

    def annotated_frame(self, frame):
        """Returns the main frame with pupils highlighted"""
        # frame = self.frame.copy()

        if self.is_located():
            color = (0, 0, 255)
            x_left, y_left = self.mouth_center_location
            x_size, y_size = self.mouth_shape
            cv2.rectangle(
                frame,
                (int(x_left-x_size/2), int(y_left-y_size/2)),
                (int(x_left+x_size/2), int(y_left+y_size/2)),
                color
            )

        return frame

    def draw_vect(self, frame):
        if self.mouth_center_location:
            point = self.mouth_center_location
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
