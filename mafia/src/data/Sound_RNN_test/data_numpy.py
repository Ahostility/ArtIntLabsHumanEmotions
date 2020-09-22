
from ...dirs import iterfiles, DIR_DATA_PROCESSED, DIR_DATA_RAW

def create_dataset_mfcc(path_dir: dir, dataframe: pd ):

    # Возвращает датасет для последующей разбивки
    
    features = []
    classes = []
    df = dataframe.loc[:, ['ID', 'CLASS']].to_numpy()
    for filename, class_maf in df:

        true_name = str(filename) + '.wav'
        input_name = os.path.join(path_dir, true_name)

        if os.path.exists(input_name):    
            print('%s.....OK' % true_name)
        else:
            print('not exist %s' % input_name)
            continue
        
        x, sr = librosa.load( input_name )
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
        mfcc = np.array(mfcc)            
        mfcc.resize((13,1600))

        features.append(mfcc.T)
        classes.append(class_maf)

    return np.asarray(features), np.array(classes)


if __name__ == '__main__':
    import librosa 
    import matplotlib.pyplot as plt 
    import librosa.display
    import os
    import numpy as np
    import pandas as pd

    # ! CREATE DATASET
    # import data_numpy

    path_dir='../../../../../data/interhim/AIMAF/audio'
    features_numpy, classes_numpy = create_dataset_mfcc(path_dir= path_dir, dataframe= df)

    #SAVE AND LOAD OUR DATA
    np.save('features_T', features_numpy)
    np.save('classes_T', classes_numpy)



    