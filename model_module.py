import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import onnxruntime as ort
import tf2onnx

# 🚀 开启混合精度计算
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

def build_optimized_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def build_tcn_model(input_shape):
    """
    🚀 TCN 模型，高度敏感于 300 分钟内变化
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, dilation_rate=1, activation='relu', input_shape=input_shape),
        Conv1D(filters=32, kernel_size=3, dilation_rate=2, activation='relu'),
        Flatten(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def convert_to_onnx(model, filename="model.onnx"):
    spec = (tf.TensorSpec((None, 20, 9), tf.float32),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

def run_onnx_inference(onnx_model_path, input_data):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: input_data})
    return pred

def calculate_advanced_score(df):
    latest = df.iloc[-1]
    vwap_score = (latest['close'] - df['VWAP'].iloc[-1]) / df['VWAP'].iloc[-1] * 10 if 'VWAP' in df else 0
    obv_score = 10 if df['OBV'].iloc[-1] > df['OBV'].iloc[-10] else -10 if 'OBV' in df and len(df) >= 10 else 0
    atr_mean = df['ATR'].mean() if 'ATR' in df else 0
    atr_score = max(min((df['ATR'].iloc[-1] - atr_mean) / atr_mean * 10 if atr_mean != 0 else 0, 10), -10)
    return vwap_score + obv_score + atr_score