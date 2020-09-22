from .....dirs import DIR_DATA_PROCESSED

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Text_loader(Dataset):

    def __init__(self, path=None, text=None, key = 'train'):
        super().__init__()
        self.key = key
        if self.key == 'train':
            data = pd.read_csv(DIR_DATA_PROCESSED / 'Dataset_text_1.csv')
            mackup = pd.read_csv(str(DIR_DATA_PROCESSED / 'MACKUP') + '/' + path).ID
            self.data = data.loc[data.ID.isin(mackup)]
            self.data.index = np.arange(len(self.data))

        elif self.key == 'predict':
            dict_text = {'TEXT': text}
            self.data = pd.DataFrame(dict_text, columns = ['TEXT'], index=[0])

        tokenizer_data = pd.read_csv(DIR_DATA_PROCESSED / 'Dataset_text_1.csv').TEXT
        tokenizer_data = tokenizer_data.dropna(axis='index', how='any')
        self.tokenizer = Tokenizer(num_words=3000)
        self.tokenizer.fit_on_texts(tokenizer_data)

        for i in self.data.index:
            values = str(self.data.loc[i][-1]).lower()
        sequences = self.tokenizer.texts_to_sequences(values)
        sample = pad_sequences(sequences, maxlen=60)
        for i in self.data.index:
            self.data.TEXT[i] = sample[i]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.key == 'train':        
            sample = self.data.loc[index].tolist()
            target = torch.tensor(sample[1])
            values = torch.tensor(sample[2])
            target = target.long()
            values = values.float()

            return values, target

        elif self.key == 'predict':
            sample = self.data.iloc[index].TEXT
            sequences = self.tokenizer.texts_to_sequences([str(sample), ' '])
            sample = pad_sequences(sequences, maxlen=60)
            return torch.LongTensor(sample[0])