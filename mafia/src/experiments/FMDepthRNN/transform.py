import torch
import numpy as np

class Normalize(object):

    def __call__(self, sample):
        sample = sample.copy().astype(np.float)
        sample -= (sample.min(axis=0) + sample.max(axis=0)) / 2
        sample /= np.sqrt((sample[:, :2] ** 2).sum(axis=1)).max()
        return sample
