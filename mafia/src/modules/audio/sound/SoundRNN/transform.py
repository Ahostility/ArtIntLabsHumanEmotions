import librosa
import numpy as np
import torch

def create_dataset_mfcc(path_file: str):

    # Возвращает датасет для последующей разбивки
    
    
    x, sr = librosa.load( path_file )
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
    mfcc = np.array(mfcc)            
    mfcc.resize((13,1600))
    mfcc = mfcc.T

    #NORMALIZE
    mean = np.mean(mfcc)
    std = np.std(mfcc)
    norm_mfcc = abs(mfcc - mean)/std
    #END NORMALIZE

    outputs = torch.FloatTensor(norm_mfcc)
    outputs = outputs.view(-1, 1600, 13)

    return outputs

