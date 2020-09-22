def Silence_Audio(path):   

    import librosa


    audio_data = path
    x , sr = librosa.load(audio_data)
    time = round(len(x)/22000, 4)

    payz = []
    mini_payz = []
    j = 0 
    for i in x:
        if i < 0.03 and i > -0.03:
            mini_payz.append(j)
        else:
            if len(mini_payz) > 11000:
                payz.append(mini_payz[0])
                payz.append(mini_payz[-1])
                mini_payz = []
            else:
                mini_payz = []
        j += 1

    payz = [round(i/22000, 2) for i in payz]

    i = 0
    payz_time = []
    while i < len(payz):
        tim = payz[i + 1] - payz[i]
        tim = round(tim, 2)
        payz_time.append(tim)
        i += 2

    little_pauze = 0
    middle_pauze = 0
    big_pauze = 0
    sum_pauses = 0

    for j in payz_time:
        if j < 1:
            little_pauze += 1
        elif j < 2:
            middle_pauze += 1
        elif j >= 2:
            big_pauze += 1
        sum_pauses += j
        
    byf = [little_pauze, middle_pauze, big_pauze, round(sum_pauses, 2), time]
    return byf


if __name__ == '__main__': Silence_Audio()