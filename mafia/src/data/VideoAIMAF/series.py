from ...modules.video.evaluate import VideoParameters
from ...dirs import DIR_DATA_INTERHIM
dataset = 'WEBAIMAF'

if __name__ == '__main__':

    from datetime import datetime
    from tqdm import tqdm

    def tprint(*args, **kwargs):
        return print('\n', str(datetime.time(datetime.now()))[:8], *args, **kwargs)

    data_dir = DIR_DATA_INTERHIM / dataset / 'video'
    savedir = DIR_DATA_INTERHIM / ('Video' + dataset)
    savedir.mkdir(parents=True, exist_ok=True)
    pattern = '*.mp4'
    for filepath in tqdm(data_dir.glob(pattern)):
        savepath = savedir / (filepath.stem + '.csv')
        filepath = filepath.as_posix()
        if not savepath.exists():
            tprint('Start:', filepath)
            params = VideoParameters()
            params.mining(filepath, 3, fps=25)
            params.save(savepath.as_posix())
            tprint('Saved:', savepath)
        else:
            tprint('Skip:', filepath)
