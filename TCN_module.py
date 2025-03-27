import os
import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client
from data_module import get_historical_data
from model_module import build_tcn_model
from logger_setup import get_logger
from indicators_module import calculate_optimized_indicators

# API 密钥和配置
API_KEY = "JdDbn4SbVDYmtvO6XzFFGtxfVxIzzb2c1Zg0HcJW6PvdOjD0Nxg03sCIUWZQ0W5a"
API_SECRET = "qnYFpJAVlbVrKibIETeuN3I35YSeDfY2UJow1GxwkxarubdRNsETkg8rpOhqX5eP"
client = Client(API_KEY, API_SECRET)

# 创建模型保存目录
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"已创建模型保存目录: {model_dir}")


# 获取有效交易对
def get_valid_futures_symbols(client):
    try:
        exchange_info = client.futures_exchange_info()
        valid_symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if
                         symbol['contractType'] == 'PERPETUAL']
        return valid_symbols
    except Exception as e:
        print(f"获取有效交易对失败: {e}")
        return []


# 获取并过滤交易对
symbols = get_valid_futures_symbols(client)
print(f"有效交易对: {symbols}")

# 仅使用已验证的交易对
config_symbols = ["ETHUSDC", "DOGEUSDC", "BNBUSDC", "SOLUSDC"]  # 仅保留已验证的交易对
symbols = [s for s in config_symbols if s in symbols]
if not symbols:
    print("没有可用的有效交易对")
    exit(1)

# 初始化训练数据
X_train, y_train = [], []

# ... 前面的导入和初始化保持不变 ...

# ... 前面的导入和初始化保持不变 ...

# 收集训练数据
X_train, y_train = [], []
mean_features, std_features = None, None

for symbol in symbols:
    df = get_historical_data(client, symbol)
    if df is None or len(df) < 30:
        print(f"跳过 {symbol}，数据不足（需要至少 30 条记录，当前 {len(df) if df is not None else 0} 条）")
        continue

    df = calculate_optimized_indicators(df)
    required_cols = ['open', 'high', 'low', 'close', 'VWAP', 'MACD', 'RSI', 'OBV', 'ATR']
    if not all(col in df.columns for col in required_cols):
        print(f"跳过 {symbol}，指标计算失败")
        continue

    df_subset = df[required_cols].iloc[-200:].ffill().fillna(0)
    features = df_subset.values

    # 标准化特征
    if mean_features is None and std_features is None:
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0) + 1e-10
    features = (features - mean_features) / std_features

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        print(f"跳过 {symbol}，特征数据异常: {features}")
        continue

    for i in range(len(features) - 10):
        if i + 19 >= len(df):
            break
        window = features[i:i + 10]
        current_price = df['close'].iloc[i + 9]
        future_price = df['close'].iloc[i + 19]  # 15 分钟后
        label = 1 if future_price > current_price * 1.005 else 0
        X_train.append(window)
        y_train.append(label)

if len(X_train) > 0 and len(y_train) > 0:
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 保存均值和标准差到文件
    np.save(os.path.join(model_dir, "mean_features.npy"), mean_features)
    np.save(os.path.join(model_dir, "std_features.npy"), std_features)

    # 构建并训练模型
    model = build_tcn_model((10, 9))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)
    print(f"训练历史 - 最终损失: {history.history['loss'][-1]}, 最终准确率: {history.history['accuracy'][-1]}")

    # 保存模型
    model_path = os.path.join(model_dir, "tcn_model.weights.h5")
    model.save_weights(model_path)
    print(f"TCN 模型已训练并保存为 {model_path}")
else:
    print("训练数据不足，无法训练模型")