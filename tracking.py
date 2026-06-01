import cv2
import numpy as np

class PanTiltTracker:

    def __init__(self):

        self.pan = 90
        self.tilt = 90

        self.last_cx = 320
        self.last_cy = 320

        self.KP = 0.1

        self.MAX_STEP = 3

        self.HFOV = 62
        self.VFOV = 48

    def update(self, cx, cy):

        cx = int(
            0.7*self.last_cx +
            0.3*cx
        )

        cy = int(
            0.7*self.last_cy +
            0.3*cy
        )

        self.last_cx = cx
        self.last_cy = cy

        error_x = cx - 320
        error_y = cy - 320

        if abs(error_x) < 10:
            error_x = 0

        if abs(error_y) < 10:
            error_y = 0

        angle_x = error_x * (
            self.HFOV / 640
        )

        angle_y = error_y * (
            self.VFOV / 640
        )

        delta_pan = angle_x * self.KP
        delta_tilt = angle_y * self.KP

        delta_pan = np.clip(
            delta_pan,
            -self.MAX_STEP,
            self.MAX_STEP
        )

        delta_tilt = np.clip(
            delta_tilt,
            -self.MAX_STEP,
            self.MAX_STEP
        )

        self.pan += delta_pan
        self.tilt += delta_tilt

        self.pan = np.clip(
            self.pan,
            0,
            180
        )

        self.tilt = np.clip(
            self.tilt,
            0,
            180
        )

        return (
            int(self.pan),
            int(self.tilt)
        )