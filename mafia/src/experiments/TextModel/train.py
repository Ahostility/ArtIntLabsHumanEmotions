from .....dirs import DIR_DATA_LOGS


if __name__ == '__main__':

    from .model import NLPMOdel
    from .dataset import Text_loader
    from .callbacks import MeterMetricsCallback, AUCCallback, PrecisionRecallF1ScoreCallback

    from datetime import datetime

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from catalyst.dl import SupervisedRunner
    from catalyst.dl.callbacks import AccuracyCallback, ConfusionMatrixCallback#, AUCCallback, PrecisionRecallF1ScoreCallback


    batch_size = 6
    num_workers = 2

    max_news_len = 50
    num_words = 3000

    epochs = 300
    
    train_dataset = Text_loader('global.csv')
    valid_dataset = Text_loader('valid.csv')
    test_dataset = Text_loader('test.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NLPMOdel(hidden_dim=90, layer_dim=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    logdir = str(DIR_DATA_LOGS / 'text') + '/' +  datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        AccuracyCallback(num_classes=2),
        AUCCallback(),
        PrecisionRecallF1ScoreCallback(),
        ConfusionMatrixCallback(num_classes=2, class_names=['peace', 'mafia'])
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=epochs,
        verbose=True
    )