# Create numpy spectogram form wav file with need second size

import librosa
import numpy as np

# IT'S MEL-SPECTOGRAM
def create_mel_spectogam(path_file, second_number=15):

    waveform, sample_rate = librosa.load(path=path_file, sr=22050 )

    waveform_size = sample_rate * second_number
    waveform = np.array(waveform)
    waveform.resize((waveform_size,))

    D = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=256))**2
    specgram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, S=D)
    specgram = np.reshape(specgram[:,:1275], (-1, 128, 25))

    return specgram

# MEL-SPEC TO WAV
def decoder_mel_spectogram(spect, path_output_file, sample_rate=22050):

    spec = np.reshape(spect, (128, 1275))
    result_waveform = librosa.feature.inverse.mel_to_audio(spec)

    librosa.output.write_wav(path_output_file, result_waveform, sample_rate)

#  IT'S STFT
def create_spectogam(path_file, second_number=15):

    waveform, sample_rate = librosa.load(path=path_file, sr=22050 )

    waveform_size = sample_rate * second_number
    waveform = np.array(waveform)
    waveform.resize((waveform_size,))

    specgram = librosa.stft(waveform, n_fft=1024, hop_length=256, dtype=float)
    specgram = np.reshape(specgram[:,:1275], (-1, 513, 25))

    return specgram

# STFT TO WAV
def decoder_spectogram(spect, path_output_file, sample_rate=22050):

    spec = np.reshape(spect, (513, 1275))
    result_waveform = librosa.istft(spec)

    librosa.output.write_wav(path_output_file, result_waveform, sample_rate)