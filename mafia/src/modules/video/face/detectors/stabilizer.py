import cv2
import numpy as np

from .dist import distance

def point_btwn(p1, p2):
    return (p1 + p2) // 2

def get_stabilize_points(eyes: np.ndarray, rectangle: np.ndarray, bottom: np.ndarray):
    leye, reye = eyes
    left, right = leye[0], reye[3]
    x, y, w, h = rectangle
    top = np.array([x + w // 2, y])
    center = point_btwn(left, right)
    return top, center, bottom, left, right

def angle_btwn_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    a = distance(p1, p2)
    b = distance(p3, p2)
    c = distance(p3, p1)
    return np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

def rotate_point(origin: np.ndarray, point: np.ndarray, angle: int):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return (qx - 1, qy - 1)

def is_point_btwn(point: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    c1 = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
    c2 = (p3[0] - p2[0]) * (point[1] - p2[1]) - (p3[1] - p2[1]) * (point[0] - p2[0])
    c3 = (p1[0] - p3[0]) * (point[1] - p3[1]) - (p1[1] - p3[1]) * (point[0] - p3[0])
    return (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0)

def stabilize(stabilize_points: tuple):
    top, center, bottom, left, right = stabilize_points
    angle = angle_btwn_points(center, bottom, top)
    checkpoint = rotate_point(bottom, center, angle)
    isin = is_point_btwn(checkpoint, bottom, center, top)
    return np.degrees((-1) ** int(isin) * angle)

def image_rotate(image: np.ndarray, angle: float, expand: bool = False):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    if expand:
        cos, sin = np.abs(m[0, 0]), np.abs(m[0, 1])
        nw, nh = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        m[0, 2] += (nw / 2) - cx
        m[1, 2] += (nh / 2) - cy
        expanded = cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC)
        return cv2.resize(expanded, (w, h))
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC)

class Stabilizer():

    def __init__(self):
        self.drop()

    def drop(self):
        self._angle = 0

    def stabilize(self, eyes: np.ndarray, rectangle: np.ndarray, bottom: np.ndarray):
        stabilize_points = get_stabilize_points(eyes, rectangle, bottom)
        self._angle += stabilize(stabilize_points)

    def rotate(self, image: np.ndarray):
        return image_rotate(image, self._angle)
