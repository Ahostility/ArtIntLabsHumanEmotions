def predict_sample(text):
    from .BERT_model.prediction import predict_bert
    import torch.nn.functional as F


    result = predict_bert(text).detach()
    result = F.softmax(result)
    
    target = result[0].argmax().tolist()
    class_names = ['peace', 'mafia']
    name = class_names[target]
    prob = result[0][target].cpu().numpy()
    output = {'text': (name, prob)}
    
    return output

def text(path):
    from .Audio_to_text import audio_to_text
    text = audio_to_text.conver(path)
    output = predict_sample(text)
    return output

def write_result(output):
    with open('text_out.txt', 'w') as f:
        for key, (name, prob) in output.items():
            f.write(f'{key},{name},{prob}')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    result = text(args.path)
    print(result)
    write_result(result)
