from typing import Union

import torchvision
import numpy as np

class BIOMETRY(torchvision.datasets.DatasetFolder):

    def __init__(self, *args, **kwargs):
        kwargs['loader'] = self.loader
        kwargs['extensions'] = ('npy',)
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index: Union[int, list]) -> dict:
        sample, target = super().__getitem__(index)
        return sample, target

    def loader(self, filepath):
        return np.load(filepath)
