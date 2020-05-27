import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

"""

Function : autoencoder
Definition: this function does the following:
            1) takes the processed dataset and split into teraining data and testing data
            2) Creates a autoencoder network for the purpose of Dimensionality reduction
            3) Train the data with the healp of the Autoencoder Network
            4) Validated the model with the test data
            
            
Return: encoded train and encoded test

"""
def autoencoder():
    x =  pd.read_csv("dataset_preprocessed_1.csv")
    #x= x.iloc[1:]
    x= x.drop(["Unnamed: 0"], axis=1)
    print(x.shape)
    print(x.head())
    encoding_dim = 10
    ncol = x.shape[1]
    X_train, X_test = x[:1000],x[1001:]

    print(X_train.shape, X_test.shape)
    input_dim = Input(shape = (ncol, ))

    encoded1 = Dense(800, activation = 'relu')(input_dim)
    #encoded11= Dense(650, activation='relu')(encoded1)
    encoded2 = Dense(500, activation = 'relu')(encoded1)
    encoded3 = Dense(300, activation = 'relu')(encoded2)
    #encoded31 = Dense(200, activation = 'relu')(encoded3)
    encoded4 = Dense(100, activation = 'relu')(encoded3)
    #encoded5 = Dense(50, activation = 'relu')(encoded4)
    encoded6 = Dense(30, activation = 'relu')(encoded4)
    encoded7 = Dense(encoding_dim, activation = 'relu')(encoded6)

    # Decoder Layers
    decoded1 = Dense(30, activation = 'relu')(encoded7)
    #decoded2 = Dense(50, activation = 'relu')(decoded1)
    decoded3 = Dense(100, activation = 'relu')(decoded1)
    #decoded31 = Dense(200, activation = 'relu')(decoded3)
    decoded4 = Dense(300, activation = 'relu')(decoded3)
    decoded5 = Dense(500, activation = 'relu')(decoded4)
    #decoded51= Dense(650, activation='relu')(decoded5)
    decoded6 = Dense(800, activation = 'relu')(decoded5)
    decoded7 = Dense(ncol, activation = 'sigmoid')(decoded6)


    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs = input_dim, outputs = decoded7)

    # Compile the Model
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    autoencoder.fit(X_train, X_train, nb_epoch = 30, batch_size = 10, shuffle = False, validation_data = (X_test, X_test))
    autoencoder.save_weights("./ae_weights.h5")
    #plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
    encoder = Model(inputs = input_dim, outputs = encoded7)
    encoded_input = Input(shape = (encoding_dim, ))

    encoded_train = pd.DataFrame(encoder.predict(X_train))
    encoded_train = encoded_train.add_prefix('feature_')

    encoded_test = pd.DataFrame(encoder.predict(X_test))
    encoded_test = encoded_test.add_prefix('feature_')
    print(encoded_train.head())
    print(encoded_test.head())

    encoded_train.to_csv('train_encoded.csv', index=False)
    encoded_test.to_csv('test_encoded.csv', index=False)

    return encoded_train, encoded_test

