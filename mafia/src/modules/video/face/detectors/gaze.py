import cv2
import numpy as np

def image_cut(imgray: np.ndarray, eyemarks: np.ndarray):
    x, y, w, h = cv2.boundingRect(eyemarks)
    return imgray[y:y+h, x:x+w]

def image_match(im1: np.ndarray, im2: np.ndarray, increase: int = 1):
    lh, lw = im1.shape[:2]
    rh, rw = im2.shape[:2]
    f = lambda x: int(x * increase)
    size = (f(max(lw, rw)), f(max(lh, rh)))
    img12 = cv2.resize(im1, size)
    img22 = cv2.resize(im2, size)
    return (img12, img22)

def image_combine(im1: np.ndarray, im2: np.ndarray):
    im1s, im2s = image_match(im1.astype(np.uint16), im2.astype(np.uint16))
    return ((im1s + im2s) // 2).astype(np.uint8)

def rad_btwn_points(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray):
    rad = np.arctan2(y1 - y0, x1 - x0)
    if rad < 0:
        rad += 2 * np.pi
    return rad

def get_pupil_position(image_gray_eye: np.ndarray, blur: tuple = (5, 5), threshold: int = 40):
    imblur = cv2.GaussianBlur(image_gray_eye, (7, 7), 0)
    _, imthreshold = cv2.threshold(imblur, threshold, 255, cv2.THRESH_BINARY_INV)
    ctr = cv2.findContours(imthreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = ctr
    if len(contours) == 0:
        return (0, 0)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    px, py, pw, ph = cv2.boundingRect(contours[0])
    x, y = px + pw // 2, py + ph // 2
    return x, y

def gaze_direction(image_gray_eye: np.ndarray):
    px, py = get_pupil_position(image_gray_eye)
    h, w = image_gray_eye.shape[:2]
    cx, cy = px + w // 2, py + h // 2
    rad = rad_btwn_points(cx, cy, px, py)
    return rad#np.degrees(rad)

# def make_cyclop(image: np.ndarray, eyes: np.ndarray):
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     imleye = image_cut(image_gray, eyes[0])
#     imreye = image_cut(image_gray, eyes[1])
#     combine = image_combine(imleye, imreye)
#     return combine

# def get_largest(eyes: np.ndarray):
#     leye, reye = eyes
#     lsize = leye[:, 1].max() - leye[:, 1].min()
#     rsize = reye[:, 1].max() - reye[:, 1].min()
#     if rsize > lsize:
#         return 1
#     else:
#         return 0

def cut_eye(image: np.ndarray, eye: np.ndarray):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cut = image_cut(image_gray, eye)
    return cut

class Gaze(object):

    def __init__(self):
        self.eye = None

    def detect(self, image: np.ndarray, eyes: np.ndarray):
        # large = get_largest(eyes)
        self.eye = cut_eye(image, eyes[0])

    def direction(self):
        return gaze_direction(self.eye)
