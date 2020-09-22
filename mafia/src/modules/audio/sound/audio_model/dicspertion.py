import librosa
import numpy as np


def extracting_metafeatich(path):
    amplitudes, _ = librosa.load(path)

    record_voice = [i**2 for i in amplitudes]
    valid_values = []
    for index, i in enumerate(record_voice):
        if i > 0.00001:
            valid_values.append(index)

    valid_voice = [amplitudes[i] for i in valid_values]

    mackup = len(valid_voice)//20
    start = 0
    end = mackup
    valid_voice_mackup = []
    for i in range(20):
        valid_voice_mackup.append(np.abs(valid_voice[start:end]))
        start += mackup
        end += mackup

    Mean_values = {}
    Max_values = {}
    Dis_values = {}
    head_mean = []
    head_max = []
    head_dis = []


    for index, fragment in enumerate(valid_voice_mackup):
        head_mean.append('head' + str(index))
        head_max.append('max' + str(index))
        head_dis.append('dis' + str(index))

        Mean_values.update({head_mean[index]: np.mean(fragment)})
        Max_values.update({head_max[index]: np.max(fragment)})
        Dis_values.update({head_dis[index]: np.var(fragment, ddof=1)})

    header = head_mean + head_max + head_dis
    data = {**Mean_values, **Max_values, **Dis_values}

    return data, header