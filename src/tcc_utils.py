import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_loss(history):
    history_df = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    history_df.plot(ylim=(0, history_df.values.max()))
    plt.title(f'Loss: {history.history["loss"][-1]}')


def plot_loss_with_validation(history):
    history_df = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    history_df.plot(ylim=(0, history_df.values.max()))
    plt.title(f'Loss: {history.history["loss"][-1]} / Val_loss: {history.history["val_loss"][-1]}')


def linear_model_weighs_table(linear_model, inputs_df):
    linear_wdf = pd.DataFrame(linear_model.get_weights()[0].T, columns=inputs_df.columns)\
        .T.sort_values(0, ascending=False)
    linear_wdf.columns = ['feature_weight']
    return linear_wdf


def calculate_diff_percent(a, b):
    return ((100 * (b - a) / b)**2)**.5


def make_array_with_error(x, y):
    array = np.array([])
    for i in range(len(x)):
        array = np.append([array], [[calculate_diff_percent(x[i], y[i])]])
    return array


def prediction_results_data_frame(x, y, model, y_scaler):
    predictions_real = y_scaler.inverse_transform(model.predict(x))
    y_real = y_scaler.inverse_transform(y)
    error = make_array_with_error(predictions_real, y_real)
    df = {'Prediction': predictions_real[:,0], 'Real': y_real[:,0], '% error': error}
    return pd.DataFrame(data=df)