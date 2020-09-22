import numpy as np

from .models import EmoNet
from .dist import distance
from .....dirs import DIR_DATA_MODELS, DIR_DATA_NUMPY

biometry_connections = (DIR_DATA_NUMPY / 'biometry_connections.npy').as_posix()
emonet = (DIR_DATA_MODELS / 'emonet.pth').as_posix()

class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

connections = np.load(biometry_connections)
model = EmoNet(emonet)

def get_biometry(pts3d, rectangle):
    w, h = rectangle[-2:]
    return np.array([distance(pts3d[i], pts3d[j]) for i, j in connections]) / np.sqrt(w * h)

class Emotions(object):

    def __init__(self):
        self.primary = None
        self.biometry = None
        self.items = {}

    def __getitem__(self, index: str):
        return self.items.get(index)

    def detect(self, pts3d: np.ndarray, rectangle: np.ndarray):
        biometry = get_biometry(pts3d, rectangle)
        predicts = model(biometry)
        index_prim = predicts.argmax()

        self.biometry = biometry
        self.primary = {class_names[index_prim]: predicts[index_prim]}
        self.items = dict(zip(class_names, predicts))
