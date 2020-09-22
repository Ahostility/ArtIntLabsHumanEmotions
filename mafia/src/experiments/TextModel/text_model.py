from ...dirs import DIR_DATA_RAW, DIR_DATA_PROCESSED, DIR_DATA_MODELS, DIR_DATA_LOGS


if __name__ == '__main__':

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from datetime import datetime


    import pandas as pd
    import numpy as np

    max_news_len = 30
    num_words = 3000
    epochs = 800

    data = pd.read_csv(DIR_DATA_PROCESSED / 'Dataset_text.csv')
    text = data['2']

    y_train = data['1'].values

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text)
    tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(text)
    x_train = pad_sequences(sequences, maxlen=max_news_len)
    
    logdir = str(DIR_DATA_LOGS / 'text' / datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq=epochs)  

    model = Sequential()
    model.add(Embedding(num_words, 64, input_length=max_news_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adamax', 
                loss='binary_crossentropy', 
                metrics=['AUC'])

    history = model.fit(x_train, 
                        y_train, 
                        epochs=epochs,
                        batch_size=12,
                        validation_split=0.1,
                        callbacks=[tensorboard_callback])

    # plt.plot(history.history['auc'], 
    #             label='Доля верных ответов на обучающем наборе')
    # plt.plot(history.history['val_auc'], 
    #             label='Доля верных ответов на проверочном наборе')
    # plt.xlabel('Эпоха обучения')
    # plt.ylabel('Доля верных ответов')
    # plt.legend()
    # plt.show()

    model.save(DIR_DATA_MODELS / 'model_lstm.h5')