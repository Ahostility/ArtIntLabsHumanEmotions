from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os

class Sound_RNN_test(torch.utils.data.Dataset):

    # def __init__(self, path_features: str, path_classes: str, train: bool, train_size: float = 0.8):
    def __init__(self, path_dir: str, train: bool, train_size: float = 0.8):

        path_classes, path_features = sorted(os.listdir(path_dir))

        features_numpy = np.load(path_dir + '/' + path_features)
        classes_numpy = np.load(path_dir + '/' + path_classes)
        if (train):
            start = None
            end = int(len(classes_numpy)*train_size)
        else:
            start = int(len(classes_numpy)*train_size)
            end = None

        #NORMALIZE
        mean = np.mean(features_numpy, axis=0)
        std = np.std(features_numpy, axis=0)
        features_numpy = abs(features_numpy - mean)/std
        #END NORMALIZE

        self.features_numpy = features_numpy[start:end]
        self.classes_numpy = classes_numpy[start:end]

        
    def __len__(self):
        return len(self.classes_numpy)

    def __getitem__(self, index: int):

        inputs = torch.FloatTensor(self.features_numpy)[index] 
        targets = torch.LongTensor(self.classes_numpy)[index]
        return inputs, targets 
    
        
