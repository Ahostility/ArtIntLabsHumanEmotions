from ...dirs import DIR_DATA_PROCESSED, DIR_DATA_LOGS

if __name__ == '__main__':

    from ..args import get_parser
    # Getting basic parameters
    parser = get_parser()
    # Add additional parameters received from the console
    parser.add_argument('--hidden_size', default=256, type=int)
    params = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms
    from catalyst.dl import SupervisedRunner
    from catalyst.dl.utils import set_global_seed, prepare_cudnn
    from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, PrecisionRecallF1ScoreCallback

    from .dataset import Sound_RNN_test
    from .model import AudioRNN

    from sklearn.model_selection import train_test_split
    import numpy as np

    # Seed & CUDA deterministic
    set_global_seed(params.seed)
    prepare_cudnn(deterministic=params.deterministic)

    # Init custom dataset
    datadir = str(DIR_DATA_PROCESSED / 'Sound_RNN_test')
    # data/processed/Sound_RNN_test/features_T.npy
    trainset = Sound_RNN_test(path_dir= datadir, train= True, train_size= 0.8)
    validset = Sound_RNN_test(path_dir= datadir, train= False, train_size= 0.8)

    # Init data loaders
    num_workers = params.num_workers
    batch_size = params.batch_size
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Init train components 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AudioRNN( n_neurons=params.hidden_size)
    # model = AudioRNN(batch_size= batch_size, n_neurons=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Init catalyst components
    runner = SupervisedRunner(device=device)
    loaders = {'train': trainloader, 'valid': validloader}
    logdir = str(DIR_DATA_LOGS / params.logdir)
    callbacks = [
        AccuracyCallback(num_classes = 2),
        AUCCallback(num_classes = 2),
        PrecisionRecallF1ScoreCallback(num_classes = 2),
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=params.num_epochs,
        verbose=params.verbose,        
    )    
    