from ...dirs import DIR_DATA_LOGS


if __name__ == '__main__': 

    from .model import GenderModel
    from .dataset import Gender_loader

    from datetime import datetime

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from catalyst.dl import SupervisedRunner
    from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, ConfusionMatrixCallback, F1ScoreCallback


    batch_size = 4
    num_workers = 2

    epochs = 50

    train_dataset = Gender_loader()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GenderModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader}
    logdir = str(DIR_DATA_LOGS / 'audio') + '/gender/' +  datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        AccuracyCallback(),
        F1ScoreCallback(),
        ConfusionMatrixCallback(num_classes=2, class_names=['female', 'male'])
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=epochs,
        # verbose=True
    )
