import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def calculate_advanced_score(df) -> float:
    """
    利用 ARIMA 模型和简单移动平均计算高级得分。
    """
    try:
        arma_model = ARIMA(df['close'], order=(2, 0, 2), enforce_stationarity=False, enforce_invertibility=False)
        arma_fit = arma_model.fit()
        arma_forecast = arma_fit.forecast(steps=1)[0]
        arma_score = (arma_forecast - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
    except Exception:
        arma_score = 0.0
    # 暂时使用 ARIMA 的结果作为 ARMAX 部分
    armax_score = arma_score
    returns = df['close'].pct_change().dropna()
    narch_score = np.mean(np.sqrt(np.abs(returns))) * 100
    sma = df['close'].rolling(window=20).mean().iloc[-1]
    current_price = df['close'].iloc[-1]
    ma_score = (current_price - sma) / sma * 100
    advanced_score = 0.4 * arma_score + 0.2 * armax_score + 0.2 * narch_score + 0.2 * ma_score
    return advanced_score

def calculate_advanced_score(df):
    # 调用 indicators_module 中的高级评分函数（简单复用）
    from indicators_module import calculate_advanced_score as adv_score
    return adv_score(df)

def calculate_garch_volatility(df) -> float:
    """
    不使用 GARCH 模型，返回 0.0 作为占位。
    """
    return 0.0
