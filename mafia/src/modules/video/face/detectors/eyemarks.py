import numpy as np

from .dist import distance

def EAR(eye: np.ndarray):
    p1, p2, p3, p4, p5, p6 = eye
    d1 = distance(p2, p6)
    d2 = distance(p3, p5)
    d3 = distance(p1, p4)
    return (d1 + d2) / (2 * d3)

class Eyemarks(object):

    def __init__(self):
        self.eyes = None

    def detect(self, pts2d: np.ndarray):
        self.eyes = np.array([pts2d[36:42], pts2d[42:48]])

    def EAR(self):
        """Eye Aspect Ratio"""
        leye, reye = self.eyes
        return (EAR(leye) + EAR(reye)) * 0.5
