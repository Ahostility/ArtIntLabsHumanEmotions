from ...dirs import DIR_DATA_MODELS, DIR_DATA_INTERHIM

if __name__ == '__main__':

    import torch
    from .model import AudioRNN
    from .transform import create_dataset_mfcc

    model = AudioRNN()

    checkpoint = torch.load(str(DIR_DATA_MODELS / 'SoundMFCCParams.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    path_file = str(DIR_DATA_INTERHIM / 'AIMAF/audio/0.wav')
    inputs = create_dataset_mfcc(path_file)
    print(torch.max(model(inputs), 1)[1].data)
    print(torch.max(model(inputs), 1)[0].data)