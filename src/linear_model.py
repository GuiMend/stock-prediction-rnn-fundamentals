# -*- coding: utf-8 -*-
import tcc_utils
import tratamento_dados_empresa
from keras.initializers import RandomUniform as rnd_uni

# Test with company: Grendene (GRND3)
x_csv_path = '../data/grendene-indicadores-financeiros-raw.csv'
y_csv_path = '../data/grendene-cotacao-raw.csv'

X = tratamento_dados_empresa.treat_economatica_indicadores_financeiros(x_csv_path)

# Spliting test / train and Feature Scaling
x_train, x_test, y_train, y_test, x_scaler, y_scaler = tratamento_dados_empresa. \
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


