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

# Compile keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# add "verbose=0" so that it doesn't print the epochs and doesn't crash in jupyter
# model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# evaluate keral model
loss, accuracy = model.evaluate(X,y)
print('Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]

# make class predictions with the model
predictions = model.predict_classes(X)

for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# summarize layers
print(model.summary())
# plot graph
from keras.utils import plot_model
plot_model(model, to_file='multilayer_perceptron_graph.png')