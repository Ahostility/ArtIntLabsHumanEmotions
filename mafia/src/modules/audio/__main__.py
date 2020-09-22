def audio(path):
    from .text.__main__ import text
    from .sound.__main__ import sound

    result = text(path)
    result.update(sound(path))

    return result

def write_result(output):
    with open('text_out.txt', 'w') as f:
        for key, (name, prob) in output.items():
            f.write(f'{key},{name},{prob}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()

    result = audio(args.path)
    print(result)
    write_result(result)
