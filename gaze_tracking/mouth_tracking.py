from __future__ import division
import cv2
from .mouth import Mouth


class MouthTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.mouth = None

    @property
    def mouth_located(self):
        """Check that the pupils have been located"""
        try:
            return len(self.mouth.mouth_center_location) == 2 and self.mouth.mouth_shape is not None
        except Exception:
            return False

    def _analyze(self, landmarks_fullface):
        if landmarks_fullface is None:
            self.mouth = None
            return

        # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        try:
            landmarks = landmarks_fullface
            self.mouth = Mouth(None, landmarks)

        except IndexError as e:
            print("ERROR: %s" % e)
            self.mouth = None

    def refresh(self, frame, landmarks_fullface):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze(landmarks_fullface)

    def is_half_open(self):
        """Returns true if the user is looking to the right"""
        return self.mouth.is_half_open()

    def is_full_open(self):
        return self.mouth.is_full_open()

    def is_closed(self):
        return self.mouth.is_closed()

    def annotated_frame(self, frame):
        """Returns the main frame with pupils highlighted"""
        # frame = self.frame.copy()

        if self.mouth_located:
            color = (0, 0, 255)
            x_left, y_left = self.mouth.mouth_center_location
            x_size, y_size = self.mouth.mouth_shape
            cv2.rectangle(
                frame,
                (int(x_left-x_size/2), int(y_left-y_size/2)),
                (int(x_left+x_size/2), int(y_left+y_size/2)),
                color
            )

        return frame
