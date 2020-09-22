import numpy as np

from .detectors import Facemarks, Eyemarks, Emotions, Gaze, Stabilizer

class FaceCapture(object):

    def __init__(self):
        self.facemarks = Facemarks()
        self.eyemarks = Eyemarks()
        self.emotions = Emotions()
        self.gaze = Gaze()
        self.stabilizer = Stabilizer()
        self.is_detected = False

    def detect(self, image: np.ndarray, stabilize: bool = False):
        if stabilize:
            image = self.stabilizer.rotate(image)

        self.is_detected = self.facemarks.detect(image)
        if self.is_detected:
            self.eyemarks.detect(self.facemarks['2D'])
            self.emotions.detect(self.facemarks['3D'], self.facemarks.rectangle)
            self.gaze.detect(image, self.eyemarks.eyes)
            bottom = self.facemarks['2D'][8]
            self.stabilizer.stabilize(self.eyemarks.eyes, self.facemarks.rectangle, bottom)
        else:
            self.stabilizer.drop()
        return self.is_detected
