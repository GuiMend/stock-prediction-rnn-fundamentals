# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tratamento_dados_empresa
import tcc_utils

import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

X = tratamento_dados_empresa.treat_economatica_indicadores_financeiros(
        '../data/grendene-indicadores-financeiros-raw.csv')

Y = tratamento_dados_empresa.treat_economatica_stock_with_following_month_opening_price(
        '../data/grendene-cotacao-raw.csv')

Y = Y.iloc[len(Y) - len(X) - 1: len(Y) - 1,:]
y = pd.DataFrame(Y['Abertura proximo mes'])


# Spliting test / train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling X
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Feature Scaling y
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# Keras Initializer with seed
glorot_normal = keras.initializers.glorot_normal(seed=0)
glorot_uniform = keras.initializers.glorot_uniform(seed=0)
random_uniform = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=0)

# First model: Linear Regression
linr_model = Sequential()
linr_model.add(Dense(1, input_shape=(X_train.shape[1],), kernel_initializer=random_uniform))

linr_model.compile('adam', 'mean_squared_error')
linr_history = linr_model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
tcc_utils.plot_loss(linr_history)

linr_train_eval = linr_model.evaluate(X_train, y_train, verbose=0)
print(f'MSE of training set using Linear model: {linr_train_eval}')
linr_test_eval = linr_model.evaluate(X_test, y_test, verbose=0)
print(f'MSE of testing set using Linear model: {linr_test_eval}')

# weights data frame
print(f'Linear model weights: {tcc_utils.linear_model_weighs_table(linr_model, X)}')


prediction_results = tcc_utils.prediction_results_data_frame(X_test, y_test, linr_model, ss_y)