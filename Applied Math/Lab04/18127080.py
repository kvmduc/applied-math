import numpy as np
import pandas as pd
from sklearn.model_selection import *
from operator import itemgetter
from tensorflow.keras import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


name_col = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

def LR_all_feature(df):
    train, test = train_test_split(df,test_size=0.1,shuffle = True)
    feature = train.to_numpy()[:, 0:-1]
    label = train.to_numpy()[:, -1:]

    test_feature = test.to_numpy()[:, 0:-1]
    test_label = test.to_numpy()[:, -1:]

    x0 = np.ones((feature.shape[0],1))
    feature = np.concatenate((x0,feature), axis = 1)
    x_hat = np.linalg.pinv(feature) @ label
    score = (np.linalg.norm(test_feature @ x_hat[1:,:] + x_hat[0:1,:] - test_label) ** 2)
    print(x_hat)
    print(score)
    return x_hat


def LR_one_feature(df):
    gen = (each for each in name_col if each != 'quality')
    data_score = []
    X = []
    kf = KFold(n_splits=10)
    for train_index, validation_index in kf.split(df):
        train_data = df.iloc[train_index]
        valid_data = df.iloc[validation_index]
        for each in gen:
            temp_train = train_data.loc[:,[each,'quality']]
            x0 = np.ones((temp_train.shape[0], 1))
            one_feature_data = np.concatenate((x0,temp_train),axis = 1)
            A_train = one_feature_data[:, 0:-1]
            label_train = one_feature_data[:, -1:]
            x_hat = np.linalg.pinv(A_train) @ label_train
            X.append(x_hat)

            temp_valid = valid_data.loc[:, [each, 'quality']]
            A_valid = temp_valid.loc[:,[each]]
            label_valid = temp_valid.loc[:,['quality']]
            A_valid = A_valid.to_numpy()
            label_valid = label_valid.to_numpy()
            data_score.append(np.linalg.norm(A_valid @ x_hat[1:,:] + x_hat[0:1,:] -  label_valid) ** 2)
    min_index = min(enumerate(data_score), key=itemgetter(1))[0]
    print(X)
    print(data_score)
    print(data_score[min_index])
    return (name_col[min_index], X[min_index])

def autoencoder(df):
    feature = df.to_numpy()[:, 0:-1]
    label = df.to_numpy()[:, -1:]
    input_data = Input(shape=(11,))
    encoded = Dense(6, activation='relu')(input_data)
    decoded = Dense(11, activation=None)(encoded)

    autoencoder = Model(input_data, decoded)
    opt = optimizers.Adam(learning_rate=1e-3)
    autoencoder.compile(optimizer=opt, loss='mse')
    callback = callbacks.EarlyStopping(monitor='loss', mode = 'min')
    autoencoder.fit(feature, feature,
                    epochs = 500,
                    shuffle=False,
                    callbacks=callback)
    bottle_neck = K.function([autoencoder.layers[0].input],[autoencoder.layers[1].output])
    layer_output = bottle_neck([feature])

    train_data, test_data = train_test_split(layer_output[0],test_size=0.1,shuffle=False)
    train_label, test_label = train_test_split(label,test_size=0.1,shuffle=False)
    x0 = np.ones((train_data.shape[0], 1))
    A_train = np.concatenate((x0,train_data),axis = 1)
    x_hat = np.linalg.pinv(A_train)@ train_label
    score = (np.linalg.norm(test_data @ x_hat[1:,:] + x_hat[0:1,:] - test_label) ** 2)
    print(score)
    print(x_hat)



if __name__ == '__main__':
    df = pd.read_csv('wine.csv',sep = ';')
    data = df.to_numpy()
    feature = df.to_numpy()[:,0:-1]
    label = df.to_numpy()[:,-1:]
    # LR_all_feature(df)
    # LR_one_feature(df)
    autoencoder(df)

