import pandas as pd
import numpy as np
import os
from .columns import *
from .upload import MARKUP_PATH, WEB_MARKUP_PATH

from moviepy.video import VideoClip
from moviepy.audio import AudioClip
from moviepy.video.io.VideoFileClip import VideoFileClip

class _MarkupSeries(pd.Series):

    @property
    def _constructor(self):
        return _MarkupSeries

    @property
    def _constructor_expanddim(self):
        return _MarkupFrame

    def subinfo(self, video_dir: str, file_pattern: str) -> tuple:
        """Provides information for cropping.

        Args:
            video_dir: The path to the directory with the video files.
            file_pattern: Abstract file name. Specify the difference as {}.

        Returns:
            The path to the file, the right and left end of the cut.
        """

        ext = os.path.splitext(file_pattern)[1]
        uniq = file_pattern.format(self.name)
        filenames = np.array(os.listdir(video_dir), dtype=np.str)
        filenames = filenames[np.char.rfind(filenames, ext) != -1]
        contain = filenames[np.char.rfind(filenames, uniq) != -1]
        # if not contain: print(file_pattern, self.name)
        filename = str(contain[0])
        filepath = os.path.join(video_dir, filename)
        start, end = self[START], self[END]
        end = end if end != 0 else None
        return filepath, start, end

class _MarkupFrame(pd.DataFrame):

    @property
    def _constructor(self):
        return _MarkupFrame

    @property
    def _constructor_sliced(self):
        return _MarkupSeries

    def by_rating(self, video: int = 0, audio: int = 0):
        f"""Query filtered data by `{RATING_VIDEO}` and/or `{RATING_AUDIO}`` columns.

        Args:
            video: Threshold sort by `{RATING_VIDEO}`. Default: 0.
            audio: Threshold sort by `{RATING_AUDIO}`. Default: 0.

        Returns:
            Filtered copy of the table.
        """

        return self.query(f'{RATING_VIDEO} >= {video} and {RATING_AUDIO} >= {audio}')

    def iter_subinfo(self, video_dir: str, file_pattern: str):
        """An generator of information for cropping. See more: ._MarkupSeries.subinfo
        """

        for index, row in self.iterrows():
            yield (index,) + row.subinfo(video_dir, file_pattern)

class MafiaDataFrame(_MarkupFrame):
    """Copy of pandas DateFrame. Contains table from data/origin_markup.csv.
    """

    def __init__(self):
        super().__init__(pd.read_csv(MARKUP_PATH, index_col='ID'))

class WebMafiaDataFrame(_MarkupFrame):
    """Copy of pandas DateFrame. Contains table from data/web_origin_markup.csv.
    """

    def __init__(self):
        super().__init__(pd.read_csv(WEB_MARKUP_PATH, index_col='ID'))
