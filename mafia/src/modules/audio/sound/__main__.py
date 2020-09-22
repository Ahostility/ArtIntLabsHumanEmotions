def get_age_gender(path):
    from .age_model import predict_model
    from .gender_model import predict_tree
    from .Pauses import count_pauses
    from .audio_model.dicspertion import extracting_metafeatich

    import os

    age = predict_model.main(path)
    gender = predict_tree.main(path)
    pauses = count_pauses.Silence_Audio(path)
    dis, header = extracting_metafeatich(path)
    
    data = make_dataframe(age, gender, pauses, dis, header)
    return data

def make_dataframe(age, gender, pauses, dis, header_dis):
    import pandas as pd
    import itertools as it
    
    header_val = ['age', 'gender', 'lit_pause', 'mid_pause', 'big_pause', 'time_pause']
    header = list(it.chain(header_val, header_dis))

    dict_values = {'age': age, 'gender': gender, 'lit_pause': pauses[0], 'mid_pause': pauses[1], 'big_pause': pauses[2], 'time_pause': pauses[3]}
    values = {**dict_values, **dis}

    data = pd.DataFrame(values, columns = header, index=[0])
    return data


def predict_sample(data):
    from .audio_model.prediction import prediction_model
    import torch.nn.functional as F

    result = prediction_model(data).detach()
    result = F.softmax(result)

    target = result.argmax().tolist()
    class_names = ['peace', 'mafia']
    name = class_names[target]
    prob = result[0][target].numpy()
    output = {'sound': (name, prob)}

    return output

def sound(path):
    data = get_age_gender(path)
    output = predict_sample(data)
    return output

def write_result(output):
    with open('sound_out.txt', 'w') as f:
        for key, (name, prob) in output.items():
            f.write(f'{key},{name},{prob}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    result = sound(args.path)
    print(result)
    write_result(result)
