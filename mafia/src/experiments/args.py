from argparse import ArgumentParser

def get_parser() -> ArgumentParser:
    """Returns an ArgumentParser with basic parameters for training model.
    """

    parser = ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--verbose', action='store_true')
    return parser
