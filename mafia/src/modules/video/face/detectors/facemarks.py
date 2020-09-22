import dlib
import numpy as np

from .models import FMDepthRNN

from .....dirs import DIR_DATA_MODELS

shape_predictor_68_face_landmarks = (DIR_DATA_MODELS / 'shape_predictor_68_face_landmarks.dat').as_posix()
fmdepthrnn = (DIR_DATA_MODELS / 'fmdepthrnn.pth').as_posix()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)
model = FMDepthRNN(fmdepthrnn)

sort_rects = lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top())

def rect_to_box(rect):
    x, y = rect.left(), rect.top()
    return np.array([x, y, rect.right() - x, rect.bottom() - y])

def shape_to_np(shape: dlib.full_object_detection):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

def normalize(y: np.ndarray):
    x = y.copy().astype(np.float32)
    # Centering
    x -= (x.min(axis=0) + x.max(axis=0)) / 2
    # Normalize
    x /= np.sqrt((x[:, :2] ** 2).sum(axis=1)).max()
    return x

class Facemarks(object):

    def __init__(self):
        self._items = {}
        self._rectangle = (0, 0, 0, 0)
        self.is_detected = False

    def __getitem__(self, index: str):
        return self._items.get(index)

    @property
    def rectangle(self):
        return self._rectangle

    def detect(self, image: np.ndarray):
        detects = sorted(detector(image), key=sort_rects, reverse=True)
        self.is_detected = bool(detects)
        if self.is_detected:
            rect = detects[0]
            landmarks = predictor(image, rect)
            pts2d = shape_to_np(landmarks)
            norm2d = normalize(pts2d)
            depth = model(norm2d)
            pts3d = np.vstack((norm2d.T, depth)).T
            self._rectangle = rect_to_box(rect)
            self._items['2D'] = pts2d
            self._items['3D'] = pts3d
        return self.is_detected
