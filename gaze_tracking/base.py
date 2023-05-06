import cv2
import numpy as np

class BaseFaceFeature(object):
    SCALE = 0.25
    def __init__(self, vect_calc, dimension_calc):
        self.vect_calc = vect_calc
        self.dimension_calc = dimension_calc

        self.feature_reference_point = None
        self.vector = None
        self.vector_flat_projection = None

        self.display_box = [100,300]
        # direction to point the drawn flat vector.
        # I'm using constants because in order to understand the general expression
        # dynamically I'd need to shift all 3d points onto the same plan.
        # Then I'd need to rotate that entire plan in line with the screen.
        # This plan would, theoretically, stay constant so...just start with a constant.
        self._slope = None
        self.x_coeff = 1 # or -1
        self._scale = 0.3

    def set_feature_reference_point(self, point):
        if point is None:
            self.feature_reference_point = None
            self.vector = None
            self.vector_flat_projection = None
            return

        self.feature_reference_point = point
        start, end, dist = self.vect_calc.find_vector(point)
        self.vector = {
            'start': start,
            'end': end,
            'dist': dist
        }


        flattened_start = np.array(
            self.display_box,
            dtype='double'
        ).reshape(-1, 2)

        hypot = self._scale * dist
        if self._slope is None:
            flattened_end = np.array(
                (flattened_start[0][0], flattened_start[0][1] + hypot),
                dtype='double'
            ).reshape(-1, 2)
        else:
            x = ((hypot**2)/(1+self._slope**2))**(1/2) * self.x_coeff
            y = x * self._slope
            flattened_end = np.array(
                (flattened_start[0][0] + x, flattened_start[0][1] + y),
                dtype='double'
            ).reshape(-1, 2)
        self.vector_flat_projection = {
            'start': np.int32(flattened_start.reshape(-1, 2)),
            'end': np.int32(flattened_end.reshape(-1, 2)),
            'dist': dist
        }

    def draw_vect(self, frame):
        if self.feature_reference_point:
            start = self.vector_flat_projection['start']
            end = self.vector_flat_projection['end']

            frame = cv2.line(
                frame,
                tuple(start[0]),
                tuple(end[0]),
                (0, 255, 0), 2, cv2.LINE_AA
            )

        return frame
