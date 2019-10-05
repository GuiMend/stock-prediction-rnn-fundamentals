# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tratamento_dados_empresa
import tcc_utils

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


# First model: Linear Regression
linr_model = Sequential()
linr_model.add(Dense(1, input_shape=(X_train.shape[1],)))

linr_model.compile('adam', 'mean_squared_error')
linr_history = linr_model.fit(X_train, y_train, epochs=30, validation_split=0.2)
tcc_utils.plot_loss(linr_history)

linr_model.evaluate(X_train, y_train, verbose=0)
linr_model.evaluate(X_test, y_test, verbose=0)

# weights data frame
linr_wdf = pd.DataFrame(linr_model.get_weights()[0].T, columns=X.columns).T.sort_values(0, ascending=False)
linr_wdf.columns = ['feature_weight']
linr_wdf.iloc[:,:]

def calc_err(a, b):
    return ((100*(b-a)/b)**2)**.5


predictions = linr_model.predict(X_test)
predictions_real = ss_y.inverse_transform(predictions)
stock_real = ss_y.inverse_transform(y_test)

for i in range(10):
	print('Predicted: %.2f vs %.2f (expected) = %.2f%% error' % (predictions_real[i], stock_real[i], calc_err(predictions_real[i], stock_real[i])))


