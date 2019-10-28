# -*- coding: utf-8 -*-
import tcc_utils
import tratamento_dados_empresa
from keras.initializers import glorot_normal as gl_normal, glorot_uniform as gl_uni, RandomUniform as rnd_uni


x_csv_path = '../data/grendene-indicadores-financeiros-raw.csv'
y_csv_path = '../data/grendene-cotacao-raw.csv'


X = tratamento_dados_empresa.treat_economatica_indicadores_financeiros(x_csv_path)


# Spliting test / train and Feature Scaling
x_train, x_test, y_train, y_test, x_scaler, y_scaler = tratamento_dados_empresa.\
    get_scaled_splits_and_scaler(x_csv_path, y_csv_path, 0.2, 0)


''' 
  1) Linear Model without val split

  80% train / 20% test / 0% validation split
  kernel_initializer: random_uniform
  layers: input(X columns), output (1)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
random_uniform = rnd_uni(minval=-0.05, maxval=0.05, seed=0)
# Create model
linear_model = tcc_utils.linear_model(x_train, random_uniform)
# Fit model
linear_history = linear_model.fit(x_train, y_train, epochs=50, verbose=0)
# Evaluate model
tcc_utils.evaluate_model(linear_model, linear_history, x_train, 
                         y_train, x_test, y_test, X, y_scaler, linear=True)


'''
  2) Linear Model with val split
  
  72% train / 20% test / 8% validation split
  kernel_initializer: random_uniform
  layers: input(X columns), output (1)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
random_uniform = rnd_uni(minval=-0.05, maxval=0.05, seed=0)
# Create model
linear_model_with_val = tcc_utils.linear_model(x_train, random_uniform)
# Fit model
linear_history_with_val = linear_model_with_val.fit(x_train, y_train, epochs=50,
                                                    validation_split=0.1, verbose=0)
# Evaluate model
tcc_utils.evaluate_model(linear_model_with_val, linear_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler, val=True, linear=True)


''' 
  3) 1st Deep model

  80% train / 20% test / 0% validation split
  kernel_initializer: random_uniform
  layers: input(X columns), 32, 16, 8, output(1)
  activations: input(none), 'relu', 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
random_uniform = rnd_uni(minval=-0.05, maxval=0.05, seed=0)
# Neural Network layers
layers=[32, 16, 8]
activations=['relu', 'relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x_train, random_uniform, layers, activations)
# Fit model
deep_history = deep_model.fit(x_train, y_train, epochs=100, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train, 
                         x_test, y_test, X, y_scaler)


''' 
  4) 2nd Deep model

  72% train / 20% test / 8% validation split
  kernel_initializer: random_uniform
  layers: input(X columns), 32, 16, 8, output(1)
  activations: input(none), 'relu', 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
random_uniform = rnd_uni(minval=-0.05, maxval=0.05, seed=0)
# Neural Network layers
layers=[32, 16, 8]
activations=['relu', 'relu', 'relu']
# Create model
deep_model_with_val = tcc_utils.deep_model(x_train, random_uniform, layers, activations)
# Fit model
deep_history_with_val = deep_model_with_val.fit(x_train, y_train, epochs=100, 
                                       validation_split=0.2, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model_with_val, deep_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler)


''' 
  5) 3rd Deep model

  80% train / 20% test / 0% validation split
  kernel_initializer: glorot_normal
  layers: input(X columns), 32, 16, 8, output(1)
  activations: input(none), 'relu', 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_normal = gl_normal(seed=0)
# Neural Network layers
layers=[32, 16, 8]
activations=['relu', 'relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x_train, glorot_normal, layers, activations)
# Fit model
deep_history = deep_model.fit(x_train, y_train, epochs=40, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train, 
                         x_test, y_test, X, y_scaler)


''' 
  6) 4th Deep model

  72% train / 20% test / 8% validation split
  kernel_initializer: glorot_normal
  layers: input(X columns), 32, 16, 8, output(1)
  activations: input(none), 'relu', 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_normal = gl_normal(seed=0)
# Neural Network layers
layers=[32, 16, 8]
activations=['relu', 'relu', 'relu']
# Create model
deep_model_with_val = tcc_utils.deep_model(x_train, glorot_normal, layers, activations)
# Fit model
deep_history_with_val = deep_model_with_val.fit(x_train, y_train, epochs=60, 
                                       validation_split=0.2, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model_with_val, deep_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler)



''' 
  7) 5th Deep model

  80% train / 20% test / 0% validation split
  kernel_initializer: glorot_uniform
  layers: input(X columns), 32, 16, 8, output(1)
  activations: input(none), 'relu', 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_uniform = gl_uni(seed=0)
# Neural Network layers
layers=[32, 16, 8]
activations=['relu', 'relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x_train, glorot_uniform, layers, activations)
# Fit model
deep_history = deep_model.fit(x_train, y_train, epochs=40, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train, 
                         x_test, y_test, X, y_scaler)


''' 
  8) 6th Deep model

  72% train / 20% test / 8% validation split
  kernel_initializer: glorot_uniform
  layers: input(X columns), 32, 16, 8, output(1)
  activations: input(none), 'relu', 'relu', 'relu', output(none)
  optimizer: adam
  loss: MSE
'''
# Keras Initializer with seed
glorot_uniform = gl_uni(seed=0)
# Neural Network layers
layers=[32, 16, 8]
activations=['relu', 'relu', 'relu']
# Create model
deep_model_with_val = tcc_utils.deep_model(x_train, glorot_uniform, layers, activations)
# Fit model
deep_history_with_val = deep_model_with_val.fit(x_train, y_train, epochs=40, 
                                       validation_split=0.2, verbose=0)
# Evaluating model
tcc_utils.evaluate_model(deep_model_with_val, deep_history_with_val, x_train,
                         y_train, x_test, y_test, X, y_scaler)








 # Leave one Out => model.train_on_batch



