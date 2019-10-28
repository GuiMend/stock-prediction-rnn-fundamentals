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


results = np.array([])

for i in range(x.shape[0]):
    x_train = np.delete(x, i, 0)
    y_train = np.delete(y, i, 0)
    x_test = x[i:i+1]
    y_test = y[i:i+1]
    deep_model.train_on_batch(x_train, y_train)
    predictions_real = y_scaler.inverse_transform(deep_model.predict(x_test))
    y_real = y_scaler.inverse_transform(y_test)
    results = np.append([results], [[tcc_utils.calculate_diff_percent(predictions_real[0,0], y_real[0,0])]])
    print(f'R$ {predictions_real[0,0]} - R$ {y_real[0,0]} - error: {tcc_utils.calculate_diff_percent(predictions_real[0,0], y_real[0,0])}')

# Evaluating model
tcc_utils.evaluate_model(deep_model, deep_history, x_train, y_train,
                         x_test, y_test, X, y_scaler)




# -----------------------
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

last_3_x = x.iloc[x.shape[0] - 3:, :]
last_3_y = y.iloc[y.shape[0] - 3:, :]
x = x.iloc[:x.shape[0] - 3,:]
y = y.iloc[:y.shape[0] - 3,:]

# Feature Scaling X
x_scaler = StandardScaler()
x = x_scaler.fit_transform(x)
last_3_x = x_scaler.transform(last_3_x)
# Feature Scaling y
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)
last_3_y = y_scaler.transform(last_3_y)
# Keras Initializer with seed
glorot_normal = gl_normal(seed=0)
# Neural Network layers
layers = [number_hidden_layer, number_hidden_layer]
activations = ['relu', 'relu']
# Create model
deep_model = tcc_utils.deep_model(x, glorot_normal, layers, activations)


results = np.array([])

for i in range(x.shape[0]):
    x_train = np.delete(x, i, 0)
    y_train = np.delete(y, i, 0)
    x_test = x[i:i+1]
    y_test = y[i:i+1]
    deep_model.train_on_batch(x_train, y_train)
    predictions_real = y_scaler.inverse_transform(deep_model.predict(x_test))
    y_real = y_scaler.inverse_transform(y_test)
    results = np.append([results], [[tcc_utils.calculate_diff_percent(predictions_real[0,0], y_real[0,0])]])
    print(f'R$ {predictions_real[0,0]} - R$ {y_real[0,0]} - error: {tcc_utils.calculate_diff_percent(predictions_real[0,0], y_real[0,0])}')


import matplotlib.pyplot as plt
import pandas as pd
plt.plot(y_scaler.inverse_transform(y),results, 'ro')
plt.show()

results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
plt.boxplot(results_df.iloc[:,0], vert=False)
plt.title(f'Error - Median: {results_df.iloc[:,0].median()}%')
plt.show()
results_df.describe()

prediction_results = tcc_utils.prediction_results_data_frame(last_3_x, last_3_y, deep_model, y_scaler)
sorted_pred = prediction_results.sort_values(by='Real', ascending=False)
# Print and plot results
print(sorted_pred)
plt.plot(sorted_pred['Real'], sorted_pred['% error'], 'k',
             sorted_pred['Real'], sorted_pred['% error'], 'ro')
plt.grid(True)
plt.title(f'Error - Mean: {sorted_pred["% error"].mean()}% / '
              f'Std: {sorted_pred["% error"].std()}%')
plt.xlabel('Stock price')
plt.ylabel('Error (%)')
plt.show()
plt.figure(figsize=(12, 6))
plt.boxplot(prediction_results["% error"], vert=False)
plt.title(f'Error - Median: {sorted_pred["% error"].median()}%')
plt.show()
sorted_pred['% error'].describe()


prediction_results = tcc_utils.prediction_results_data_frame(last_3_x, last_3_y, deep_model, y_scaler)
sorted_pred = prediction_results.sort_values(by='Real', ascending=False)
# Print and plot results
print(prediction_results)
plt.plot(prediction_results.index, prediction_results['% error'], 'k',
             prediction_results.index, prediction_results['% error'], 'ro')
plt.grid(True)
plt.title(f'Error - Mean: {prediction_results["% error"].mean()}% / '
              f'Std: {prediction_results["% error"].std()}%')
plt.xlabel('Stock price')
plt.ylabel('Error (%)')
plt.show()
plt.figure(figsize=(12, 6))
plt.boxplot(prediction_results["% error"], vert=False)
plt.title(f'Error - Median: {prediction_results["% error"].median()}%')
plt.show()
prediction_results['% error'].describe()

