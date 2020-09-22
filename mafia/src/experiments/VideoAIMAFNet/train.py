from ...dirs import DIR_DATA_LOGS

if __name__ == '__main__':

    from ..args import get_parser
    parser = get_parser()
    parser.add_argument('--hidden_size', default=128, type=int)
    params = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    # from catalyst.dl import SupervisedRunner
    # from catalyst.dl.utils import set_global_seed, prepare_cudnn
    # from catalyst.dl.callbacks import AccuracyCallback#, AUCCallback#, PrecisionRecallF1ScoreCallback
    # from .auc import AUCCallback

    from .dataset import VideoAIMAF
    from .model import VideoAIMAFNet

    # set_global_seed(params.seed)
    # prepare_cudnn(deterministic=params.deterministic)

    train_dataset = VideoAIMAF('train')
    valid_dataset = VideoAIMAF('valid')
    test_dataset = VideoAIMAF('test')

    num_workers = params.num_workers
    batch_size = params.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    it = iter(train_loader)
    print(next(it))
    # it = iter(valid_loader)
    # print(next(it))
    # it = iter(test_loader)
    # print(next(it))

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = VideoAIMAFNet(input_size = 7, hidden_size=params.hidden_size)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    # print('Number of parameters:', len(model))

    # runner = SupervisedRunner(device=device)
    # loaders = {'train': train_loader, 'valid': valid_loader}
    # logdir = DIR_DATA_LOGS.joinpath(params.logdir).as_posix()

    # callbacks = [
    #     AccuracyCallback(num_classes=2),
    #     AUCCallback(num_classes=2, class_names=['peace', 'mafia']),
    #     # PrecisionRecallF1ScoreCallback(num_classes=2),
    # ]

    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     callbacks=callbacks,
    #     logdir=logdir,
    #     num_epochs=params.num_epochs,
    #     verbose=params.verbose,
    #     minimize_metric=False,
    #     main_metric="auc/_mean",
    # )
