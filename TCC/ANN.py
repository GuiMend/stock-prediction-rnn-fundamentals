# -*- coding: utf-8 -*-

# Importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read output DF and transform index to datetimeindex
y = pd.read_csv('../data/grendene-cotacao.csv', index_col=0)
y.index = pd.to_datetime(y.index)

# Read inpout DF and transform index to datetimeindex
X = pd.read_csv('../data/grendene-indicadores-financeiros.csv', index_col=0)
X.index = pd.to_datetime(X.index)

# Join inputs and outputs to create the complete DF
data_df = X
data_df = data_df.join(y)

# Part 1 - Data preprocessing

# Spliting test / train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Build ANN

# Importing
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
regressor = Sequential()

# Adding input layer and hidden layer
regressor.add(Dense(29, kernel_initializer='uniform', activation='sigmoid', input_dim=56))

# Adding second hidden layer
regressor.add(Dense(29, kernel_initializer='uniform', activation='sigmoid'))

# Adding output layer
regressor.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
regressor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 3 - Making predictions and evaluating model
regressor.fit(x=X_train, y=y_train, batch_size=5, epochs=100)














