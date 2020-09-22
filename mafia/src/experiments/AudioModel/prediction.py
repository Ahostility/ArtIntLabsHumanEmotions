from .....dirs import DIR_DATA_LOGS, DIR_DATA_MODELS


def prediction_model(data):

    import torch
    from torch.utils.data import DataLoader

    from catalyst.dl import SupervisedRunner

    from .model import SoundModel
    from .dataset import Audio_loader

    model = SoundModel()
    weights_path = DIR_DATA_MODELS.joinpath('audio_model.pth')
    model.init_weights(weights_path.as_posix())
    model.eval()

    predict_dataset = Audio_loader(data=data, key='predict')
    batch_size = len(predict_dataset)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size)

    sample = next(iter(predict_loader))
    preds = model(sample)

    return preds