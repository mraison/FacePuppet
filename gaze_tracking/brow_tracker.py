from __future__ import division
import cv2
from .eye_brows import EyeBrow


class BrowTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.left_brow = None
        self.right_brow = None

    def are_raised(self):
        if self.left_brow and self.right_brow:
            return self.left_brow.is_raised() and self.right_brow.is_raised()
        return False

    def are_neutral(self):
        if self.left_brow and self.right_brow:
            return self.left_brow.is_neutral() and self.right_brow.is_neutral()
        return False

    def are_furrowed(self):
        if self.left_brow and self.right_brow:
            return self.left_brow.is_furrowed() and self.right_brow.is_furrowed()
        return False

    @property
    def brow_located(self):
        """Check that the pupils have been located"""
        try:
            return len(self.left_brow.position) == 2 and len(self.right_brow.position) == 2
        except Exception:
            return False

    def _analyze(self, landmarks_fullface):
        if landmarks_fullface is None:
            self.left_brow = None
            self.right_brow = None
            return

        # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        try:
            landmarks = landmarks_fullface
            self.left_brow = EyeBrow(self.frame, landmarks, 0)
            self.right_brow = EyeBrow(self.frame, landmarks, 1)

        except IndexError as e:
            print("ERROR: %s" % e)
            self.left_brow = None
            self.right_brow = None

    def refresh(self, frame, landmarks_fullface):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze(landmarks_fullface)

    def annotated_frame(self, frame):
        """Returns the main frame with pupils highlighted"""
        # frame = self.frame.copy()

        if self.brow_located:
            color = (255, 0, 0)
            pt1 = self.left_brow.position
            pt2 = self.right_brow.position
            cv2.line(frame, (pt1[0] - 10, pt1[1]), (pt1[0] + 10, pt1[1]), color)
            cv2.line(frame, (pt2[0] - 10, pt2[1]), (pt2[0] + 10, pt2[1]), color)

        return frame
