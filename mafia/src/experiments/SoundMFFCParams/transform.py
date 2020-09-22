import librosa
import numpy as np
import torch

def create_dataset_mfcc(path_file: str):

    # Возвращает датасет для последующей разбивки
    
    
    x, sr = librosa.load( path_file )
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
    mfcc = np.array(mfcc)            
    mfcc.resize((20,3000))

    outputs = torch.FloatTensor(mfcc.T)
    outputs = outputs.view(-1, 3000, 20)

    return outputs

