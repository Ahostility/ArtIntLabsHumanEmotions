import pandas as pd
import numpy as np
import csv

import shutil
import os

import librosa

from tqdm import tqdm


def preprocess(type_data: str, size_sample: int, DIR_DATA_RAW: str, DIR_DATA_PROCESSED: str, DIR_DATA_INTERHIM: str): 
    path = DIR_DATA_RAW / 'common-voice/'
    dict_age = {'teens':0, 'twenties':1, 'thirties':2, 'fourties':3, 'fifties':4, 'sixties':5, 'seventies':6}

    data = pd.read_csv(path / str('cv-valid-' + type_data + '.csv'))
    data = data.dropna(axis='index', how='any', subset=['age'])
    data = data[['filename', 'age']]

    path = []
    marks = []
    values_exp = size_sample * 2

    for class_age in dict_age:
        for index in data.index:
            if len(path) < values_exp and class_age == data['age'][index]:
                path.append(data['filename'][index])
                marks.append(dict_age[data['age'][index]])       
        values_exp += size_sample

    for index, mark in enumerate(marks):
        if mark == 1: marks[index] = 1
        if mark == 2: marks[index] = 1
        if mark == 3: marks[index] = 2
        if mark == 4: marks[index] = 2
        if mark == 5: marks[index] = 3
        if mark == 6: marks[index] = 3

    values = [path, marks]  

    dict_values = {}
    header = ['filename', 'class']

    for i in range(len(values)):
        dict_z = {header[i]: values[i]}
        dict_values.update(dict_z)
    dataset = pd.DataFrame(dict_values, columns=header)

    dataset.to_csv(DIR_DATA_INTERHIM / str('Age_razmetka_' + type_data + '.csv'))


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


def WavParam(DIR_DATA_RAW: str, type_data: str,DIR_DATA_PROCESSED: str, DIR_DATA_INTERHIM: str):
    DIR_DATA_RAW = DIR_DATA_RAW / 'common-voice'
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate mean sd median mode Q25 Q75 IQR skew kurt'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(DIR_DATA_PROCESSED / str('Age_dataset_' + type_data + '.csv'), 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    data = pd.read_csv(DIR_DATA_INTERHIM / str('Age_razmetka_' + type_data + '.csv'))

    for index in tqdm(range(len(data))):

        way = DIR_DATA_RAW / data['filename'][index]

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
        filename = str(way).split('cv-valid-' + type_data + '\\')[1]
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(mean)} \
                      {np.mean(sd)} {np.mean(median)} {np.mean(mode)} {np.mean(Q25)} {np.mean(Q75)} {np.mean(IQR)} {np.mean(skew)} {np.mean(kurt)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += ' ' + str(data['class'][index])
        
        file = open(DIR_DATA_PROCESSED / str('Age_dataset_' + type_data + '.csv'), 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())        