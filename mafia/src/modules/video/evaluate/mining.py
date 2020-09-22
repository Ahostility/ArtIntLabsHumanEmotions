import pandas as pd
import numpy as np

from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import VideoClip
from tqdm import tqdm
from multiprocessing import Pool

from ..face import FaceCapture

def iter_frame(clip: VideoClip, processes: int, kwargs: dict):
    captures = [FaceCapture() for i in range(processes)]
    previous_frame = None
    for index, frame in enumerate(tqdm(clip.iter_frames(**kwargs))):
        if previous_frame is not None:
            clone_value = (previous_frame == frame).astype(float).mean()
            if clone_value > 0.99:
                continue
        previous_frame = frame
        capture = captures[index % processes]
        yield index, frame, capture

def extract_frame(args):
    index, frame, capture = args
    if capture.detect(frame, stabilize=True):
        item = {}
        item.update({'blink_rate': capture.eyemarks.EAR()})
        item.update(capture.emotions.items)
        item.update({'gaze_direct': capture.gaze.direction()})
        return (index, item)

def parallel_mining(clip: VideoClip, processes: int, kwargs: dict):
    with Pool(processes) as pool:
        return pool.map(extract_frame, iter_frame(clip, processes, kwargs))

class VideoParameters(object):

    def __init__(self):
        self.reset_series()

    def reset_series(self):
        self.series = {}

    def mining(self, filepath: str, processes: int, **kwargs):
        self.reset_series()
        with VideoFileClip(filepath).without_audio() as clip:
            results = parallel_mining(clip, processes, kwargs)
        results = sorted(filter(None, results), key=lambda x: x[0])
        for index, item in results:
            for key, value in item.items():
                if self.series.get(key) is None:
                    self.series[key] = []
                self.series[key].append(value)

    def inference(self):
        ret = {}
        for key, value in self.series.items():
            ret[key] = np.var(value)
        # ret['gaze_direct'] /= np.pi
        del ret['gaze_direct']
        return ret

    def save(self, filepath: str):
        df = pd.DataFrame(self.series)
        df.to_csv(filepath, index=False)

    def load(self, filepath: str):
        df = pd.read_csv(filepath)
        series = df.to_dict()
        for key, value in series.items():
            series[key] = list(value.values())
        self.series = series
