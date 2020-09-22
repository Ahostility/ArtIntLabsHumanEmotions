import os
import torch
import scipy.io
import numpy as np

class AFLW2000(torch.utils.data.Dataset):

    def __init__(self, root_dir: str, start: int = 0, end: int = None):
        self.filepaths = np.char.add(root_dir + '/', sorted(os.listdir(root_dir))[start:end])
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        mat = scipy.io.loadmat(filepath)
        sample = mat['pt3d_68'].T
        sample = torch.FloatTensor(sample)
        inputs = sample[:, :2]
        targets =  sample[:, 2]
        return inputs, targets
