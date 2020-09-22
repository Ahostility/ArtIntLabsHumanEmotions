from ...dirs import DIR_DATA_PROCESSED, DIR_DATA_LOGS

if __name__ == '__main__':

    import torch
    from torch.utils.data import DataLoader
    from .dataset import VideoAIMAF
    from .model import VideoAIMAFNet

    model = VideoAIMAFNet(256)
    weights_path = DIR_DATA_LOGS.joinpath('VideoAIMAFNet/0/checkpoints/best.pth')
    model.init_weights(weights_path.as_posix())
    model.eval()

    data_dir = DIR_DATA_PROCESSED.joinpath('VideoWEBAIMAF').as_posix()
    test_dataset = VideoAIMAF(data_dir)
    batch_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    class_names = ['peace', 'mafia']

    sample = next(iter(test_loader))
    inputs, targets = sample
    preds = model(inputs).detach().argmax(dim=1)

    print(preds)
    # for index, (inputs, targets) in enumerate(tqdm(test_loader)):
    #     # inputs, targets = inputs[0], targets[0]
    #     preds = model(inputs).detach()#.numpy().argmax()
    #     probs = F.softmax(preds)
    #     class_num = probs.argmax(dim=0)
    #     class_name = class_names[class_num]
    #     print(class_name)
