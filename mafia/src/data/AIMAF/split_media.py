from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path
from ...dirs import DIR_DATA_RAW, DIR_DATA_INTERHIM
from ..markup.frame import WebMafiaDataFrame, MafiaDataFrame
dataset = 'WEBAIMAF'

def submedia(info):
    """Crops a media file in a separate process.
    """

    index, filepath, start, end, args = info
    try:
        with VideoFileClip(filepath).subclip(start, end) as clip:
            if end and end >= clip.duration:
                print('Bad markup:', filepath)
                end = None
            if args.video:
                videodir = DIR_DATA_INTERHIM / dataset / 'video'
                videodir.mkdir(parents=True, exist_ok=True)
                videopath = videodir / f'{index}.mp4'
                if not videopath.exists():
                    clip.write_videofile(videopath.as_posix(),
                        fps=args.fps,
                        codec='libx264',
                        verbose=args.verbose,
                        audio=False)
            if args.audio:
                audiodir = DIR_DATA_INTERHIM / dataset / 'audio'
                audiodir.mkdir(parents=True, exist_ok=True)
                audiopath = audiodir / f'{index}.wav'
                if not audiopath.exists():
                    clip.audio.write_audiofile(audiopath.as_posix(),
                        fps=args.Hz,
                        codec='pcm_s32le',
                        verbose=args.verbose)
    except OSError:
        print('Skip:', filepath)

if __name__ == '__main__':

    # Parse console arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--fps', default=None, type=int)
    parser.add_argument('--Hz', default=44100, type=int)
    parser.add_argument('--processes', default=1, type=int)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    from multiprocessing import Pool

    # Get dataframe
    df = WebMafiaDataFrame()

    # Get generator of information of cropping
    def itergen():
        for data in df.iter_subinfo(DIR_DATA_RAW / dataset, '{}.mp4'):
            yield data + (args,)

    # Download in multiprocessing
    with Pool(processes=args.processes) as pool:
        pool.map(submedia, itergen())
