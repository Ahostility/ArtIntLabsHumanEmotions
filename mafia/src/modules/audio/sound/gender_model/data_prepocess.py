import librosa 
import csv
import numpy as np
import os


def WatToWav(path):
    filenames = []
    for trac in os.listdir(path / 'female'):
        filenames.append(str(path / 'female' / trac))

    for trac in os.listdir(path / 'male'):
        filenames.append(str(path / 'male' / trac))
    return filenames


def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    result_d = {
        'mean': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt
    }

    return result_d


def WavParam(DIR_DATA_RAW, DIR_DATA_PROCESSED):
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate mean sd median mode Q25 Q75 IQR skew kurt'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(DIR_DATA_PROCESSED / 'Gender_dataset.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genders = 'female male'.split()
    
    data_path = WatToWav(DIR_DATA_RAW / 'VCTK-Corpus/wav')
    len_data_path = len(data_path)

    for index, way in enumerate(data_path):
        print(index, 'из', len_data_path)
        y, sr = librosa.load(way, mono=True, duration=30)
        rmse = librosa.feature.rmse(y=y)
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
        filename = way.split('male\\')[1]
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(mean)} \
                      {np.mean(sd)} {np.mean(median)} {np.mean(mode)} {np.mean(Q25)} {np.mean(Q75)} {np.mean(IQR)} {np.mean(skew)} {np.mean(kurt)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        if '\\male' in way: to_append += f' male'
        elif '\\female' in way: to_append += f' female'
        else: print('Error class')
        file = open(DIR_DATA_PROCESSED / 'Gender_dataset.csv', 'a', newline='')
        os.system('cls' if os.name == 'nt' else 'clear')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())    