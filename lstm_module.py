import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LSTM as OriginalLSTM
from tensorflow.keras.optimizers import Adam

# 自定义 LSTM 类，移除 time_major 参数
class LSTM(OriginalLSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def load_lstm_model(path="lstm_model.h5", input_shape=(60, 1)):
    if os.path.exists(path):
        return load_model(path, custom_objects={'LSTM': LSTM, 'mse': tf.keras.losses.MeanSquaredError()})
    else:
        return build_lstm_model(input_shape)

def online_update_lstm(model, X_train, y_train, epochs=1, batch_size=32):
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def save_lstm_model(model, path="lstm_model.h5"):
    model.save(path)

def predict_with_lstm(model, X):
    return model.predict(X)
