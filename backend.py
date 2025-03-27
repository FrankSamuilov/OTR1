from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv
import os
from indicators_module import calculate_optimized_indicators
from config import CONFIG

load_dotenv()
app = Flask(__name__)
CORS(app)  # 启用 CORS


# 模拟数据函数
def get_simulated_data(symbol):
    print(f"模拟获取 {symbol} 数据...")
    try:
        data = {
            'time': pd.date_range(end=pd.Timestamp.now(), periods=50, freq='30min'),
            'open': [100 + i * 0.5 for i in range(50)],
            'high': [101 + i * 0.5 for i in range(50)],
            'low': [99 + i * 0.5 for i in range(50)],
            'close': [100 + i * 0.5 for i in range(50)],
            'volume': [1000 + i * 10 for i in range(50)]
        }
        df = pd.DataFrame(data)
        # 模拟计算指标
        if len(df) > 0:  # 确保 DataFrame 非空
            df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df
    except Exception as e:
        print(f"Error in get_simulated_data: {e}")
        return pd.DataFrame()  # 返回空 DataFrame 以避免崩溃


@app.route('/api/trade-pairs', methods=['GET'])
def get_trade_pairs():
    print("收到 /api/trade-pairs 请求")
    return jsonify({"trade_pairs": CONFIG["TRADE_PAIRS"]})


@app.route('/api/pair/<symbol>', methods=['GET'])
def get_pair_data(symbol):
    print(f"收到 /api/pair/{symbol} 请求")
    if symbol not in CONFIG["TRADE_PAIRS"]:
        return jsonify({"error": "Unsupported trading pair"}), 400

    df = get_simulated_data(symbol)
    if df.empty:
        print(f"DataFrame is empty for {symbol}")
        return jsonify({"error": "Failed to fetch simulated data"}), 500

    df = calculate_optimized_indicators(df)
    if df.empty:
        print(f"Indicators calculation failed for {symbol}")
        return jsonify({"error": "Failed to calculate indicators"}), 500

    data = {
        "symbol": symbol,
        "historical_data": df[['time', 'open', 'high', 'low', 'close']].tail(50).to_dict(orient='records'),
        "indicators": {
            "RSI": float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not df.empty else None,
            "MACD": float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not df.empty else None,
            "VWAP": float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns and not df.empty else None
        }
    }
    return jsonify(data)


if __name__ == "__main__":
    print("启动 Flask 服务...")
    app.run(debug=True, host='0.0.0.0', port=5000)