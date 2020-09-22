# import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ...dirs import DIR_DATA_INTERHIM
from ...modules.video.evaluate import VideoParameters

class VideoAIMAF(torch.utils.data.Dataset):

    def __init__(self, stage: str):
        data_dir = DIR_DATA_INTERHIM.joinpath('VideoAIMAF')
        filepaths = list(data_dir.rglob('*.csv'))
        csvpath = Path(__file__).absolute().parent
        df = pd.read_csv(csvpath.joinpath(stage + '.csv'), index_col='ID')
        if stage == 'train':
            test = pd.read_csv(csvpath.joinpath('test.csv'), index_col='ID')
            valid = pd.read_csv(csvpath.joinpath('valid.csv'), index_col='ID')
            df = df.drop(test.index.values).drop(valid.index.values)
            del test
            del valid
        filepaths = filter(lambda x: int(x.stem) in df.index.values, filepaths)
        self.filepaths = list(filepaths)
        self.df = df

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        targets = self.df.loc[int(filepath.stem), 'CLASS']
        self.evaluate.load(filepath)
        sample = list(self.evaluate.inference().values()) + [targets]
        sample = torch.tensor(inputs)
        inputs = sample[:-1].float()
        targets = sample[-1].long()
        return inputs, targets
