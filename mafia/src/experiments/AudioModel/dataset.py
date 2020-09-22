from .....dirs import DIR_DATA_PROCESSED

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class Audio_loader(Dataset):

    def __init__(self, path=None, data=None, key = 'train'):
        super().__init__()
        self.key = key
        if self.key == 'train':
            data = pd.read_csv(DIR_DATA_PROCESSED / 'Gloabal_dataset_audio.csv')
            mackup = pd.read_csv(str(DIR_DATA_PROCESSED / 'MACKUP') + '/' + path).ID
            self.data = data.loc[data.ID.isin(mackup)]
            self.data.index = np.arange(len(self.data))

        elif self.key == 'predict':
            self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.key == 'train':        

            sample = torch.tensor(self.data.loc[index].tolist())
            target = sample[-1].long()
            values = sample[1:-1].float()

            return values, target

        elif self.key == 'predict':
            sample = torch.tensor(self.data.loc[index].tolist())
            sample = sample.float()
            return sample