import numpy as np
import cv2

class ThreeDimensionalCalc():
    DIST_COEFFS = np.zeros((4, 1))
    def __init__(
            self,
            rotation_vec,
            translation_vec,
            camera_matrix,
            frame_shape
    ):
        # We can assume we're calculating based off the tip of the nose.
        # I believe we can assume this is in the center of the screen.
        # I also don't think it's important where the nose is in 3d space really.
        # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
        # Initialize standard elements for our context:
        # What I mean is...given we're looking at a face we can assume
        # our reference point is the tip of the nose in the center of the screen
        # with a zconst of 0.0
        x = int(frame_shape[0] / 2)
        y = int(frame_shape[1] / 2)
        ref_point = np.array(
            [[x], [y], [1]], dtype="double"
        )
        z_const = 0.0
        # transform the vectors into matricies.
        self.rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        self.rotation_vec = rotation_vec
        self.translation_vec = translation_vec
        self.camera_matrix = camera_matrix

        self._rotation_matrix_inv = np.linalg.inv(self.rotation_matrix)
        self._camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        leftSideMat_part = np.dot(
            self._rotation_matrix_inv,
            self._camera_matrix_inv
        )
        self._leftSideMat = np.dot(
            leftSideMat_part,
            ref_point
        )
        self._rightSideMat = np.dot(
            np.linalg.inv(self.rotation_matrix),
            self.translation_vec
        )

        self.scale_const = z_const + (self._rightSideMat[2, 0] / self._leftSideMat[2, 0])

        # roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
        # pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
        # yaw = np.clip(np.degrees(steady_pose[0][2]), -90, 90)

    def to_3d(self, point):
        point = np.array(
            [
                [point[0]],
                [point[1]],
                [1]
            ], dtype="double"
        )
        return np.dot(
            self._rotation_matrix_inv,
            np.dot(
                self.scale_const*self._camera_matrix_inv,
                point
            ) - self.translation_vec
        )

    def to_2d(self, points):
        point_3d = np.array(points, dtype=np.float32).reshape(-1, 3)
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          self.rotation_vec,
                                          self.translation_vec,
                                          self.camera_matrix,
                                          self.DIST_COEFFS)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d

    def find_distance(self, point1, point2):
        dist_part = np.sum((point1 - point2) ** 2, axis=0)
        return np.sqrt(dist_part)

class FeatureVectorFinder(object):
    TIP_OF_NOSE = 33
    def __init__(
            self,
            landmarks,
            calculator
    ):
        self.landmarks = landmarks
        self._calculator = calculator
        start_point_2d = np.array(
            [
                self.landmarks.part(self.TIP_OF_NOSE).x, self.landmarks.part(self.TIP_OF_NOSE).y
            ], dtype=np.float32
        )
        self._start_point = self._calculator.to_3d(start_point_2d)

    def find_vector(self, point): # returns (start, end, distance)
        point_3d = self._calculator.to_3d(point)
        dist = self._calculator.find_distance(self._start_point, point_3d)
        return (self._start_point, point_3d, dist)