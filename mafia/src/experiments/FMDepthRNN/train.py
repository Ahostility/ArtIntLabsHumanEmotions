# dvc run --no-exec -f stages/FMDepthRNN_experiment0.dvc -d data/processed/AFLW2000 -o data/logs/FMDepthRNN/0/checkpoints/best.pth -o data/logs/FMDepthRNN/0/checkpoints/best_full.pth -o data/logs/FMDepthRNN/0/checkpoints/last.pth -o data/logs/FMDepthRNN/0/checkpoints/last_full.pth -o data/logs/FMDepthRNN/0/_base_log -o data/logs/FMDepthRNN/0/train_log -o data/logs/FMDepthRNN/0/valid_log -o data/logs/FMDepthRNN/0/log.txt -m -o data/logs/FMDepthRNN/0/checkpoints/_metrics.json python -m src.experiments.FMDepthRNN.train --batch_size 100 --num_epochs 200 --logdir FMDepthRNN/0 --hidden_size 136 --num_layers 1 --verbose --deterministic
from ...dirs import DIR_DATA_RAW, DIR_DATA_LOGS

if __name__ == '__main__':

    from ..args import get_parser
    # Getting basic parameters
    parser = get_parser()
    # Add additional parameters received from the console
    parser.add_argument('--hidden_size', default=None, type=int)
    parser.add_argument('--num_layers', default=None, type=int)
    params = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from catalyst.dl import SupervisedRunner
    from catalyst.dl.utils import set_global_seed, prepare_cudnn
    from catalyst.dl.callbacks import AccuracyCallback

    from .dataset import AFLW2000
    from .model import FMDepthRNN
    from .transform import Normalize

    # Seed & CUDA deterministic
    set_global_seed(params.seed)
    prepare_cudnn(deterministic=params.deterministic)

    # Init custom transforms
    transform = transforms.Compose([
        Normalize(),
    ])

    # Init custom dataset
    data_dir = DIR_DATA_RAW.joinpath('AFLW2000').as_posix()
    threshold = 1500
    train_dataset = AFLW2000(data_dir, end=threshold, transform=transform)
    valid_dataset = AFLW2000(data_dir, start=threshold, transform=transform)

    # Init data loaders
    num_workers = params.num_workers
    batch_size = params.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Init train components 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FMDepthRNN(hidden_size=params.hidden_size, num_layers=params.num_layers)
    print('Number of parameters:', len(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    criterion = torch.nn.MSELoss()

    # Init catalyst components
    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader}
    logdir = DIR_DATA_LOGS.joinpath(params.logdir).as_posix()
    callbacks = [
        AccuracyCallback(),
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
