# dvc run --no-exec -f stages/EmoNet_experiment0.dvc -d data/processed/BIOMETRY -o data/logs/EmoNet/0/checkpoints/best.pth -o data/logs/EmoNet/0/checkpoints/best_full.pth -o data/logs/EmoNet/0/checkpoints/last.pth -o data/logs/EmoNet/0/checkpoints/last_full.pth -o data/logs/EmoNet/0/_base_log -o data/logs/EmoNet/0/train_log -o data/logs/EmoNet/0/valid_log -o data/logs/EmoNet/0/log.txt -m -o data/logs/EmoNet/0/checkpoints/_metrics.json python -m src.experiments.EmoNet.train --batch_size 90 --num_epochs 100 --logdir EmoNet/0 --verbose --deterministic
from ...dirs import DIR_DATA_PROCESSED, DIR_DATA_LOGS

if __name__ == '__main__':

    # Getting basic parameters
    from ..args import get_parser
    parser = get_parser()
    # Add additional parameters received from the console
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--sample', default=0, type=int)
    params = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from catalyst.dl import SupervisedRunner
    from catalyst.dl.utils import set_global_seed, prepare_cudnn
    from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, PrecisionRecallF1ScoreCallback

    from .dataset import BIOMETRY
    from .model import *
    from .transform import Normalize, ToTensor

    # Seed & CUDA deterministic
    set_global_seed(params.seed)
    prepare_cudnn(deterministic=params.deterministic)

    # Init custom transforms
    transform = transforms.Compose([
        Normalize(params.sample == 0),
        ToTensor(),
    ])

    # Init custom dataset
    data_dir = DIR_DATA_PROCESSED.joinpath('BIOMETRY')
    traindir = data_dir.joinpath('train').as_posix()
    validdir = data_dir.joinpath('valid').as_posix()
    train_dataset = BIOMETRY(traindir, transform=transform)
    valid_dataset = BIOMETRY(traindir, transform=transform)

    # Init data loaders
    num_workers = params.num_workers
    batch_size = params.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Init train components 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = None
    if params.sample == 0:
        model = EmoNet0(hidden_size=params.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif params.sample == 1:
        model = EmoNet0(hidden_size=params.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    print('Number of parameters:', len(model))
    criterion = torch.nn.CrossEntropyLoss()

    # Init catalyst components
    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader}
    logdir = DIR_DATA_LOGS.joinpath(params.logdir).as_posix()

    class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    num_classes = 7
    # activation = 'Sigmoid'
    callbacks = [
        AccuracyCallback(num_classes=num_classes),
        # AUCCallback(num_classes=num_classes, class_names=class_names, activation=activation),
        # PrecisionRecallF1ScoreCallback(num_classes=num_classes, class_names=class_names, activation=activation),
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
