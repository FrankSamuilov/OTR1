import pandas as pd
import numpy as np

def calculate_indicators(df):
    # 计算 SMA 和 EMA
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    # 计算 MACD 与信号线
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # 计算布林带
    df['std20'] = df['close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA20'] + 2 * df['std20']
    df['Bollinger_Lower'] = df['SMA20'] - 2 * df['std20']
    # 计算 RSI (14周期)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # 随机指标 (%K)
    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - low14) / (high14 - low14))
    return df

def score_market(df):
    """
    综合评分函数：
      - 若 MACD > Signal 得 20 分，否则 0 分；
      - RSI < 30 加 10分，RSI > 70 扣 10分；
      - 收盘价低于布林带下轨加 10分，高于上轨扣 10分；
      - %K < 20 加 5分，%K > 80 扣 5分；
    最后加上基准 50 分，确保得分大致在 0～100 之间。
    """
    latest = df.iloc[-1]
    score = 0
    score += 20 if latest['MACD'] > latest['Signal'] else 0
    if latest['RSI'] < 30:
        score += 10
    elif latest['RSI'] > 70:
        score -= 10
    if latest['close'] < latest['Bollinger_Lower']:
        score += 10
    elif latest['close'] > latest['Bollinger_Upper']:
        score -= 10
    if latest['%K'] < 20:
        score += 5
    elif latest['%K'] > 80:
        score -= 5
    return score + 50

def calculate_advanced_score(df):
    # 高级评分示例：若当前价格高于 SMA20 则 +5，否则 -5
    latest = df.iloc[-1]
    return 5 if latest['close'] > latest['SMA20'] else -5
