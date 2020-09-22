from nnAudio import Spectrogram
from scipy.io import wavfile
import torch
sr, song = wavfile.read('D:/210.wav') # Loading your audio
x = song.mean(1) # Converting Stereo  to Mono
x = torch.tensor(x).float() # casting the array into a PyTorch Tensor

spec_layer = Spectrogram.STFT(n_fft=2048, freq_bins=None, hop_length=512, 
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect', 
                              fmin=50,fmax=11025, sr=sr) # Initializing the model
                              
spec = spec_layer(x)
print(spec)