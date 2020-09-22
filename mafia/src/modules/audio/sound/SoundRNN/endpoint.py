import torch
from .model import AudioRNN
from .transform import create_dataset_mfcc

def sound_module(path_file: str, path_model_checkpoint: str):

    model = AudioRNN(n_neurons= 128)

    # checkpoint = torch.load(str(DIR_DATA_MODELS / 'SoundMFCCParams.pth'))
    checkpoint = torch.load(path_model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    inputs = create_dataset_mfcc(path_file)

    return torch.argmax(model(inputs))