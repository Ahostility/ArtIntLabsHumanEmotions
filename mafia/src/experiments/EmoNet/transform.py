import torch
import numpy as np

class Normalize(object):

    def __init__(self, is_normalize):
        self.is_normalize = is_normalize

    def __call__(self, sample):
        if self.is_normalize:
            norm = 1 / np.sqrt((sample ** 2).sum())
            sample *= norm
        return sample

class ToTensor(object):

    def __call__(self, sample):
        return torch.FloatTensor(sample)
