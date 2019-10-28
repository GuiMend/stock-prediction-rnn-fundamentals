# -*- coding: utf-8 -*-
import tcc_utils
import tratamento_dados_empresa
from keras.initializers import glorot_normal as gl_normal, glorot_uniform as gl_uni, RandomUniform as rnd_uni

x_csv_path = '../data/grendene-indicadores-financeiros-raw.csv'
y_csv_path = '../data/grendene-cotacao-raw.csv'

X = tratamento_dados_empresa.treat_economatica_indicadores_financeiros(x_csv_path)

# Spliting test / train and Feature Scaling
x_train, x_test, y_train, y_test, x_scaler, y_scaler = tratamento_dados_empresa. \
    get_scaled_splits_and_scaler(x_csv_path, y_csv_path, 0.2, 0)

number_hidden_layer = round(x_train.shape[1]*1.5)

''' 
  1) Deep Feed Forward NN

  80% train / 20% test / 0% validation split
  kernel_initializer: random_uniform
  layers: input(X columns), (X columns * 1.5), (X columns * 1.5), output(1)
  activations: input(none), 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
random_uniform = rnd_uni(minval=-0.05, maxval=0.05, seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x_train, random_uniform, layers, activations)
# Fit model
deep_history = deep_model.fit(x_train, y_train, epochs=100, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train,
                         x_test, y_test, X, y_scaler)


''' 
  2) Deep Feed Forward NN with validation split

  72% train / 20% test / 8% validation split
  kernel_initializer: random_uniform
  layers: input(X columns), (X columns * 1.5), (X columns * 1.5), output(1)
  activations: input(none), 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
random_uniform = rnd_uni(minval=-0.05, maxval=0.05, seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model_with_val = tcc_utils.deep_model(x_train, random_uniform, layers, activations)
# Fit model
deep_history_with_val = deep_model_with_val.fit(x_train, y_train, epochs=100,
                                                validation_split=0.2, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model_with_val, deep_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler)


''' 
  3) Deep Feed Forward NN, with glorot normal

  80% train / 20% test / 0% validation split
  kernel_initializer: glorot_normal
  layers: input(X columns), (X columns * 1.5), (X columns * 1.5), output(1)
  activations: input(none), 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_normal = gl_normal(seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x_train, glorot_normal, layers, activations)
# Fit model
deep_history = deep_model.fit(x_train, y_train, epochs=40, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train,
                         x_test, y_test, X, y_scaler)


''' 
  4) Deep Feed Forward NN, with glorot normal and validation split

  72% train / 20% test / 8% validation split
  kernel_initializer: glorot_normal
  layers: input(X columns), (X columns * 1.5), (X columns * 1.5), output(1)
  activations: input(none), 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_normal = gl_normal(seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model_with_val = tcc_utils.deep_model(x_train, glorot_normal, layers, activations)
# Fit model
deep_history_with_val = deep_model_with_val.fit(x_train, y_train, epochs=60,
                                                validation_split=0.2, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model_with_val, deep_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler)


''' 
  5) Deep Feed Forward NN, with glorot uniform

  80% train / 20% test / 0% validation split
  kernel_initializer: glorot_uniform
  layers: input(X columns), (X columns * 1.5), (X columns * 1.5), output(1)
  activations: input(none), 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_uniform = gl_uni(seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x_train, glorot_uniform, layers, activations)
# Fit model
deep_history = deep_model.fit(x_train, y_train, epochs=40, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train,
                         x_test, y_test, X, y_scaler)

''' 
  6) Deep Feed Forward NN, with glorot uniform and validation split

  72% train / 20% test / 8% validation split
  kernel_initializer: glorot_uniform
  layers: input(X columns), (X columns * 1.5), (X columns * 1.5), output(1)
  activations: input(none), 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_uniform = gl_uni(seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model_with_val = tcc_utils.deep_model(x_train, glorot_uniform, layers, activations)
# Fit model
deep_history_with_val = deep_model_with_val.fit(x_train, y_train, epochs=60,
                                                validation_split=0.2, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model_with_val, deep_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler)
