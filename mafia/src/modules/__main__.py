def is_mafia(path):
    from .audio.text.__main__ import text
    from .audio.sound.__main__ import sound
    from .video.__main__ import video

    result = video(path)
    result.update(sound(path))
    result.update(text(path))

    return result

def write_result(output):
    with open('global_out.txt', 'w') as f:
        for key, (name, prob) in output.items():
            f.write(f'{key},{name},{prob}')



# if __name__ == '__main__':

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', default=None, type=str)
    # args = parser.parse_args()
    #
    # result = is_mafia(args.path)
    # print(result)
    # write_result(result)
