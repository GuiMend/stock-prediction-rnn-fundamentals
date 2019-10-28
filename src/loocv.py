import numpy as np
import tcc_utils
import tratamento_dados_empresa
from sklearn.preprocessing import StandardScaler
from keras.initializers import glorot_normal as gl_normal

x_csv_path = '../data/grendene-indicadores-financeiros-raw.csv'
y_csv_path = '../data/grendene-cotacao-raw.csv'

x, y = tratamento_dados_empresa.get_x_y(x_csv_path, y_csv_path)
x_int_index = x.reset_index()

number_hidden_layer = round(x.shape[1]*1.5)

# Feature Scaling X
x_scaler = StandardScaler()
x = x_scaler.fit_transform(x)
# Feature Scaling y
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)

# Keras Initializer with seed
glorot_normal = gl_normal(seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x, glorot_normal, layers, activations)


results = []

for i in range(x.shape[0]):
    x_train = np.delete(x, i, 0)
    y_train = np.delete(y, i, 0)
    x_test = x[i:i+1]
    y_test = y[i:i+1]
    deep_model.train_on_batch(x_train, y_train)
    predictions_real = y_scaler.inverse_transform(deep_model.predict(x_test))
    y_real = y_scaler.inverse_transform(y_test)
    print(f'R$ {predictions_real[0,0]} - R$ {y_real[0,0]} - error: {tcc_utils.calculate_diff_percent(predictions_real[0,0], y_real[0,0])}')

# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train,
                         x_test, y_test, X, y_scaler)