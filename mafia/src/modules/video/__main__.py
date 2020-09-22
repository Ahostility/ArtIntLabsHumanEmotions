import torch
import torch.nn.functional as F

from src.modules.video.evaluate import VideoParameters, VideoAIMAFNet
from src.dirs import DIR_DATA_MODELS

def get_video_data(path, processes, fps):
    params = VideoParameters()

    fmt = path[-3:]
    if fmt == 'csv':
        params.load(path)
    else:
        params.mining(path, processes, fps=fps)

    inputs = params.inference()
    inputs = list(inputs.values())
    inputs = torch.FloatTensor(inputs).unsqueeze(0)
    return inputs

def video(path, processes=4, fps=10):

    inputs = get_video_data(path, processes, fps)

    model = VideoAIMAFNet(1024, 0.5, 0.5)
    model.init_weights(DIR_DATA_MODELS.joinpath('videoaimafnet.pth').as_posix())
    model.eval()

    preds = model(inputs).detach().squeeze(0)
    probs = F.softmax(preds)

    target = probs.argmax().tolist()
    class_names = ['peace', 'mafia']
    name = class_names[target]
    prob = probs[target].numpy()

    output = {'video': (name, prob)}

    return output

def write_result(output):
    with open('video_out.txt', 'w') as f:
        for key, (name, prob) in output.items():
            f.write(f'{key},{name},{prob}')

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--processes', default=4, type=int)
    parser.add_argument('--fps', default=10, type=int)
    args = parser.parse_args()

    result = video(args.path, args.processes, args.fps)
    print(result)
    write_result(result)
