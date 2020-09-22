from .....dirs import DIR_DATA_RAW, DIR_DATA_MODELS, DIR_DATA_PROCESSED, DIR_DATA_INTERHIM, DIR_DATA_LOGS


def normalization_data(data):
    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate mean sd median mode Q25 Q75 IQR skew kurt'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    for column in header:
        for i in data.index:
            mean = np.mean(data[column], axis=0)
            std = np.std(data[column], axis=0)
            data[column][i] = abs(data[column][i] - mean)/std
        
    return data


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    import librosa

    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from .data_preprocess import preprocess, WavParam


    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, RNN, Dropout
    import tensorflow as tf
    from tensorflow import keras
    from datetime import datetime



    def read_csv(path: str):
        dataset = pd.read_csv(path)
        dataset = dataset.drop(['filename'], axis=1)
        data = dataset.loc[:, dataset.columns != 'label']
        marks = dataset['label']

        return data, marks

    # Преобразование датасета
    # preprocess('test', 50, DIR_DATA_RAW, DIR_DATA_PROCESSED, DIR_DATA_INTERHIM)
    # preprocess('train', 1500, DIR_DATA_RAW, DIR_DATA_PROCESSED, DIR_DATA_INTERHIM) 

    # WavParam(DIR_DATA_RAW, 'test', DIR_DATA_PROCESSED, DIR_DATA_INTERHIM)
    # WavParam(DIR_DATA_RAW, 'train', DIR_DATA_PROCESSED, DIR_DATA_INTERHIM)

    data_train, marks_train = read_csv(DIR_DATA_PROCESSED / 'Age_dataset_train.csv')
    data_test, marks_test = read_csv(DIR_DATA_PROCESSED / 'Age_dataset_test.csv')

    # data_train = normalization_data(data_train)


    # cv = StratifiedKFold()
    # forest = RandomForestClassifier()
    # forest.fit(data_train.values, marks_train.values)

    epochs = 1000

    logdir = str(DIR_DATA_LOGS / 'audio_age' /  datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq=epochs) 

    model = Sequential()
    model.add(Dense(512))
    model.add(Dense(256))    
    model.add(Dense(128))
    model.add(Dense(128))
    model.add(Dense(2, activation='sigmoid'))
 
    model.compile(optimizer='Adamax', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    print(data_train.values)

    history = model.fit(data_train.values, 
                        marks_train.values,
                        epochs=epochs,
                        batch_size=128,
                        validation_split=0.1,
                        callbacks=[tensorboard_callback])

    pickle.dump(forest, open(DIR_DATA_MODELS / 'model_age.sav', 'wb'))

    # print(np.mean(cross_val_score(forest, data.values, marks.values, cv=cv)))

    print('Result test = ', forest.score(data_test.values, marks_test.values))