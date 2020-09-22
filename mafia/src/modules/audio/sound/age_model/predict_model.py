from .....dirs import DIR_DATA_RAW, DIR_DATA_MODELS


def main(path: str):

    import pandas as pd
    import numpy as np

    import librosa

    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from .data_preprocess import spectral_properties
    

    filename = path

    y, sr = librosa.load(filename, mono=True, duration=30)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    parametrs = spectral_properties(y, sr)
    mean = parametrs['mean']
    sd = parametrs['sd']
    median = parametrs['median']
    mode = parametrs['mode']
    Q25 = parametrs['Q25']
    Q75 = parametrs['Q75']
    IQR = parametrs['IQR']
    skew = parametrs['skew']
    kurt = parametrs['kurt']
    to_append = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr), np.mean(mean),
                        np.mean(sd), np.mean(median), np.mean(mode),np.mean(Q25), np.mean(Q75),np.mean(IQR), np.mean(skew), np.mean(kurt)]            
    for e in mfcc:
        to_append.append(np.mean(e))

    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate mean sd median mode Q25 Q75 IQR skew kurt'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    dict_values = {}
    for i in range(len(header)):
        dict_z = {header[i]: to_append[i]}
        dict_values.update(dict_z)
    df = pd.DataFrame(dict_values, columns=header,  index=[0])

    forest = pickle.load(open(DIR_DATA_MODELS / 'model_age.sav', 'rb'))
    result = forest.predict(df)

    if not result: return 0
    elif result == 1: return 1
    elif result == 2: return 2
    elif result == 3: return 3


if __name__ == '__main__': main()