from ...dirs import iterfiles, DIR_DATA_PROCESSED, DIR_DATA_RAW
from ...modules.video.face import FaceCapture
from pathlib import Path
import numpy as np
import cv2

def save_biometry(args):
    capture, stage, filepath, label = args
    filename = Path(filepath).stem + '.npy'
    outdir = DIR_DATA_PROCESSED / 'BIOMETRY' / stage / label
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename
    if not outpath.exists():
        im = cv2.imread(filepath)
        if not capture.detect(im):
            return print('Cannot detect:', filepath)
        print(f'Extract {filepath} to {outpath.as_posix()}\n')
        np.save(outpath.as_posix(), capture.emotions.biometry)

if __name__ == '__main__':

    # Parse console arguments
    # python -m src.data.BIOMETRY.main --processes 4 --stepval 4
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--stepval', default=2, type=int)
    parser.add_argument('--processes', default=1, type=int)
    args = parser.parse_args()

    from itertools import chain
    from multiprocessing import Pool

    import pandas as pd

    label_names = {
        0: 'neutral',
        1: 'anger',
        2: 'contempt',
        3: 'disgust',
        4: 'fear',
        5: 'happy',
        6: 'sadness',
        7: 'surprise',
    }

    def get_cohnkanade_labeled_images():
        for filepath in iterfiles(DIR_DATA_RAW / 'CK/Emotions'):
            with open(filepath, 'r') as f:
                num = int(round(float(f.read().strip()), 2))
                label = label_names[num]
                imagedir = DIR_DATA_RAW / str(filepath.parent).replace('Emotions', 'Images')
                images = list(iterfiles(imagedir))
                yield images[-1].as_posix(), label
                yield images[-2].as_posix(), label
                if len(images) >= 9:
                    yield images[-3].as_posix(), label

    def np_str_slice(arr, start: int = None, stop: int = None, step: int = 1) -> np.ndarray:
        return np.vectorize(lambda x: x[start:stop:step], otypes=[str])(arr)

    def get_jaffe_label_images():
        datapath = DIR_DATA_RAW / 'JAFFE'
        df1 = pd.read_csv((datapath / 'SRD_FEAR_EXCLUDE.csv').as_posix(), sep=' ', index_col='#')
        df1['FEA'] = 0.0
        df2 = pd.read_csv((datapath / 'SRD.csv').as_posix(), sep=' ', index_col='#')
        df = pd.concat((df1, df2), ignore_index=True)
        df['CON'] = 0.0
        df['NEU'] = 0.0
        df['PIC'] = df['PIC'].str.replace('-','.')
        cols = np.char.upper(list(label_names.values()))
        cols = np_str_slice(cols, 0, 3).tolist()
        df = df[cols + ['PIC']]
        images = datapath / 'Images'
        for index, row in df.iterrows():
            num = np.argmax(row[cols].values, axis=0)
            label = label_names[num]
            pattern = row['PIC'] + '*.tiff'
            try:
                image = next(images.rglob(pattern))
                yield image.as_posix(), label
            except StopIteration:
                print('Not found pattern:', pattern)

    def iterchain(stepval, processes):
        captures = [FaceCapture() for i in range(processes)]
        stages = ['train', 'valid']
        for i, data in enumerate(chain(get_cohnkanade_labeled_images(), get_jaffe_label_images())):
            index_capture = i % processes
            index_stage = int(i % stepval == 0)
            yield (captures[index_capture], stages[index_stage]) + data
    
    with Pool(processes=args.processes) as pool:
        pool.map(save_biometry, iterchain(args.stepval, args.processes))
