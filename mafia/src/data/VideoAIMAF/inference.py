from ...modules.video.evaluate import VideoParameters
from ...dirs import DIR_DATA_INTERHIM, DIR_DATA_PROCESSED
from ...data.markup import MafiaDataFrame, WebMafiaDataFrame

if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    input_dir = DIR_DATA_INTERHIM.joinpath('VideoAIMAF')
    output_dir = DIR_DATA_PROCESSED.joinpath('VideoAIMAF8')
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = '*.csv'

    params = VideoParameters()
    # frame = WebMafiaDataFrame()

    frame = MafiaDataFrame().by_rating(video=8)
    peace = frame.query('CLASS == 0').head(100)
    mafia = frame.query('CLASS == 1').head(80)
    frame = pd.concat([peace, mafia])

    for filepath in input_dir.glob(pattern):
        params.load(filepath.as_posix())

        class_id = int(filepath.stem)
        if not class_id in frame.index.values:
            print('Skip:', class_id)
            continue

        # video_rate = frame.loc[class_id, 'RATING_VIDEO']
        # if video_rate < 8: continue

        class_value = frame.loc[class_id, 'CLASS']
        inf = params.inference()
        inf['class'] = int(class_value)

        arr = np.array(list(inf.values()))
        savepath = output_dir / f'{class_id}.npy'
        print('Saved:', savepath)
        np.save(savepath.as_posix(), arr)

        print(np.load(savepath.as_posix()))
