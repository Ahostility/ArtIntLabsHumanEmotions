from .....dirs import  DIR_DATA_MODELS, DIR_DATA_PROCESSED

def predict_bert(text):
    import torch
    from keras.preprocessing.sequence import pad_sequences
    import pandas as pd
    import numpy as np
    from pytorch_transformers import BertTokenizer, BertForSequenceClassification

    from .model import BERTClassifierModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print('cpu')
    else:
        n_gpu = torch.cuda.device_count()
        print(torch.cuda.get_device_name(0))


    model = BERTClassifierModel()
    checkpoint = torch.load(DIR_DATA_MODELS / 'BERTModel.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    sentences = '[CLS] ' + str(text) + ' [SEP]'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tok = tokenizer.tokenize(sentences)

    input_ids = tokenizer.convert_tokens_to_ids(tok)
    input_ids = pad_sequences(
        [input_ids, ''],
        maxlen=100,
        dtype="long",
        truncating="post",
        padding="post"
    )

    attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

    train_inputs = torch.tensor(input_ids[0]).long().to(device)
    train_masks = torch.tensor(attention_masks[0]).long().to(device)

    train_inputs = train_inputs.unsqueeze_(0)
    train_masks = train_masks.unsqueeze_(0)

    model.to(device)

    logits = model((train_inputs, train_masks))

    return logits