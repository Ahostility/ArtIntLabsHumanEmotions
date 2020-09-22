from .....dirs import DIR_DATA_LOGS


def prediction_model(text):

    import torch
    from torch.utils.data import DataLoader

    from catalyst.dl import SupervisedRunner

    from .model import NLPMOdel
    from .dataset import Text_loader

    model = NLPMOdel()
    weights_path = DIR_DATA_LOGS.joinpath('text/20200614-195729/checkpoints/best.pth')
    model.init_weights(weights_path.as_posix())
    model.eval()

    predict_dataset = Text_loader(text=text, key='predict')
    batch_size = len(predict_dataset)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size)

    sample = next(iter(predict_loader))
    preds = model(sample).detach().argmax(dim=1)
    
    return preds