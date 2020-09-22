import cv2
import numpy as np

from mpl_toolkits.mplot3d.art3d import Line3D

from ..plot import Plot
from .face import FaceCapture
from .detectors.stabilizer import get_stabilize_points
from .detectors.emotions import connections
from .detectors.gaze import cut_eye, get_pupil_position

def show_landmarks(img: np.ndarray, pts2d: np.ndarray):
    for i, j in connections:
        cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), (0, 255, 0), 1)

def show_stabilizer(img: np.ndarray, facecap: FaceCapture):
    img[:, :, :] = facecap.stabilizer.rotate(img)
    bottom = facecap.facemarks['2D'][8]
    top, center, bottom, left, right = get_stabilize_points(facecap.eyemarks.eyes, facecap.facemarks.rectangle, bottom)

    cv2.circle(img, tuple(top), 3, (255, 0, 0), -1)
    cv2.circle(img, tuple(center), 3, (255, 0, 0), -1)
    cv2.circle(img, tuple(bottom), 3, (255, 0, 0), -1)
    # cv2.circle(img, tuple(left), 3, (0, 0, 255), -1)
    # cv2.circle(img, tuple(right), 3, (0, 0, 255), -1)
    cv2.line(img, tuple(bottom), tuple(center), (255, 255, 0), 1)
    cv2.line(img, tuple(top), tuple(center), (255, 255, 0), 1)
    cv2.line(img, tuple(top), tuple(bottom), (255, 255, 0), 1)

def show_gaze(img: np.ndarray, facecap: FaceCapture):
    eyes = facecap.eyemarks.eyes
    eyemarks = eyes[0]
    imeye = cut_eye(img, eyemarks)
    px, py = get_pupil_position(imeye)
    h, w = imeye.shape[:2]
    cx, cy = w // 2, h // 2
    x, y = eyemarks[:, 0].min(), eyemarks[:, 1].min()
    w, h = img.shape[1], img.shape[0]
    cv2.circle(img, (x, y), 5, (0, 0, 255))
    cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)
    cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    # cv2.line(img, (cx, cy), (x, y), (0, 255, 255), 1)

class FacemarksPlot(Plot):

    def __init__(self, facecap: FaceCapture):
        super().__init__(projection='3D', figsize=(10, 6))
        self.facecap = facecap
        self.show_trackbars()

    def show_trackbars(self):
        na = lambda _: None
        cv2.namedWindow('Axes')
        cv2.createTrackbar('Eval' ,'Axes', 45, 180, na)
        cv2.createTrackbar('Azim' ,'Axes', 45, 180, na)

    def show2d(self, frame: np.ndarray, landmarks: bool, stabilizer: bool, gaze: bool):
        """Plot facemarks in 2D projection on frame"""
        if self.facecap.is_detected:
            if landmarks:
                pts2d = self.facecap.facemarks['2D']
                show_landmarks(frame, pts2d)
            if stabilizer:
                show_stabilizer(frame, self.facecap)
            if gaze:
                show_gaze(frame, self.facecap)
        cv2.imshow('2D Mask', frame)

    def show3d(self):
        """Plot facemarks in 3D projection"""
        self.clear()
        if self.facecap.is_detected:
            pts3d = self.facecap.facemarks['3D']
            for i, j in connections:
                p, q = pts3d[i], pts3d[j]
                line = Line3D((p[0], q[0]), (p[1], q[1]), (p[2], q[2]), c='green')
                self.ax.add_line(line)
            self.ax.scatter(*pts3d.T, c='black', linewidths=0.4, alpha=0.4)
            elev = cv2.getTrackbarPos('Eval', 'Axes')
            azim = cv2.getTrackbarPos('Azim', 'Axes')
            self.rotate_axes(elev, azim)
        self.imshow('3D Mask')

class EmotionPlot(Plot):

    def __init__(self, facecap: FaceCapture):
        super().__init__(projection='2D', figsize=(8, 4))
        self.facecap = facecap

    def show_bar(self, frame: np.ndarray):
        """Plot facemarks in 3D projection"""
        self.clear()
        if self.facecap.is_detected:
            emotions = self.facecap.emotions
            prim = emotions.primary
            key = next(iter(prim))
            text = '{}: {:.3f}'.format(key, round(prim[key], 3))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            keys = list(emotions.items.keys())
            values = list(emotions.items.values())
            self.ax.bar(keys, values, align='center', alpha=0.6)
        self.imshow('Emotions')
