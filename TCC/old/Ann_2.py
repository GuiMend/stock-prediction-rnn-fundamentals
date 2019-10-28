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
    # create model
    model = Sequential()
    model.add(Dense(56, input_dim=56, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5)
kfold = KFold(n_splits=3)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ml_model = baseline_model()
ml_model.fit(X_train,y_train,epochs=150, batch_size=10)
loss, accuracy = ml_model.evaluate(X_test,y_test)
predictions = ml_model.predict(X_test)

def calc_acc(a, b):
    return 100*(b-a)/b

for i in range(10):
	print('%.2f (expected %.2f) = %.2f accuracy' % (predictions[i, 0], y_test.iloc[i, 0], calc_acc(predictions[i], y_test.iloc[i, 0])))


import numpy as np
acc = np.array([])
acc2 = []
for i in range(10):
#    acc2.append(calc_acc(predictions[i], y_test.iloc[i, 0]))
    np.append(acc, calc_acc(predictions[i], y_test.iloc[i, 0])])

print('Mean: %.2f (%.2f)' % (acc.mean(), acc.std()))






