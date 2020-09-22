from ...dirs import DIR_DATA_PROCESSED, DIR_DATA_RAW

import torch
from torch.utils.data import Dataset

import librosa
import numpy as np
import pandas as pd
import os


def slice_into_frames(amplitudes, window_length, hop_length):
    return librosa.core.spectrum.util.frame(
        np.pad(amplitudes, int(window_length // 2), mode='reflect'),
        frame_length=window_length, hop_length=hop_length)
    # выход: [window_length, num_windows]


def get_STFT(amplitudes, window_length, hop_length):
    """ Compute short-time Fourier Transform """
    # разбиваем амплитуды на пересекающиеся фреймы [window_length, num_frames]
    frames = slice_into_frames(amplitudes, window_length, hop_length)
    
    # получаем веса для Фурье, float[window_length]
    fft_weights = librosa.core.spectrum.get_window('hann', window_length, fftbins=True)
    
    # применяем преобразование Фурье
    stft = np.fft.rfft(frames * fft_weights[:, None], axis=0)
    return stft


def get_melspectrogram(amplitudes, sample_rate=22050, n_mels=128,
                       window_length=2048, hop_length=512, fmin=1, fmax=8192):
    """
    Implement mel-spectrogram as described above.
    :param amplitudes: float [num_amplitudes]
    :param sample rate: число отсчетов каждую секунду
    :param n_mels: число каналов спектрограммы
    :param window_length: параметр размера окна для Фурье
    :param hop_length: размер пересечения 
    :param f_min: мин частота
    :param f_max: макс частота
    :returns: мел-scaled спектрограмма [n_mels, duration]
    """
    # Шаг 1
    stft = get_STFT(amplitudes, window_length, hop_length)
    assert stft.shape == (window_length // 2 + 1, len(amplitudes) // 512 + 1)
    
    # Шаг 2
    spectrogram = np.abs(stft ** 2)
    
    # Шаг 3
    mel_basis = librosa.filters.mel(sample_rate, n_fft=window_length,
                                    n_mels=n_mels, fmin=fmin, fmax=fmax)

    
    mel_spectrogram = np.dot(mel_basis, spectrogram)
    assert mel_spectrogram.shape == (n_mels, len(amplitudes) // 512 + 1)
    
    return mel_spectrogram


def preprocess_sample(amplitudes, sr=22050, max_length=150):
    spectrogram = get_melspectrogram(amplitudes, sample_rate=sr)[:, :max_length]
    spectrogram = np.pad(spectrogram, [[0, 0], [0, max(0, max_length - spectrogram.shape[1])]], mode='symmetric')
    # target = 0 if gender == 'F' else 1
    return np.array(spectrogram)#, np.int64(target)


class Gender_loader(Dataset):
    
    def __init__(self, path=None, key = 'train'):
        super().__init__()
        self.key = key
        if self.key == 'train':
            self.data_way = pd.read_csv(DIR_DATA_PROCESSED / 'mackup_gender_4.0.csv')

        elif self.key == 'predict':
            dict_path = {'filename': path}
            self.data_way = pd.DataFrame(dict_path, columns = ['filename'], index=[0])

    def __len__(self):
        return self.data_way.shape[0]

    def __getitem__(self, index):
        if self.key == 'train':  
            amplitudes, _  = librosa.load(str(DIR_DATA_RAW / 'Gender_dataset') + '/' +  self.data_way.filename[index])
            sample = preprocess_sample(amplitudes)
            # sample = sample.transpose()
            
            labels = self.data_way.label[index]
            label = np.zeros((2))
            if labels: label[1] = 1
            else: label[0] = 1   
            return torch.FloatTensor(sample), torch.Tensor(label)

        if self.key == 'predict':
            amplitudes, _  = librosa.load(self.data_way.filename[index])
            sample = preprocess_sample(amplitudes)
            # sample = sample.transpose()
            return torch.LongTensor(sample)