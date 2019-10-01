# -*- coding: utf-8 -*-

# Imports
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read output DF and transform index to datetimeindex
y = pd.read_csv('../data/grendene-cotacao.csv', index_col=0)
y.index = pd.to_datetime(y.index)

# Read inpout DF and transform index to datetimeindex
X = pd.read_csv('../data/grendene-indicadores-financeiros.csv', index_col=0)
X.index = pd.to_datetime(X.index)

def baseline_model():
    model = Sequential()
    model.add(Dense(56, input_shape=(,56)))