from .....dirs import DIR_DATA_RAW, DIR_DATA_PROCESSED


def main(text):
    import pandas as pd
    import numpy as np


    dict_nigativ = [' не ', ' ни ']

    result = []
    for index, sample in enumerate(text):
        result.append(0)
        for i in dict_nigativ:
            if i in sample: result[index] = 1

    
    
    return result


if __name__ == '__main__': main()