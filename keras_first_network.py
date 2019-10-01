# -*- coding: utf-8 -*-

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into inputs (X) and outputs (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# define keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
