CONFIG = {
    "TRADE_PAIRS": ["ETHUSDC", "DOGEUSDC", "BNBUSDC", "SOLUSDC", "VINEUSDT", "TSTUSDT", "KAITOUSDT", "AIXBTUSDT"],
    "TRADE_PERCENTAGE": 0.5,          # 可用期货余额比例
    "MAX_LEVERAGE": 20,
    "MIN_NOTIONAL": 23,             # 最低下单金额，23 USDC
    "OBSERVATION_PERIOD": 300,      # 观察期时长（秒）
    "OBSERVATION_INTERVAL": 30,     # 观察间隔（秒）
    "OBSERVATION_EXTRA_SCORE": 5,   # 观察期目标评分增量
    "LSTM_UPDATE_THRESHOLD": 10,    # LSTM 在线更新时缓冲区样本数
    "MIN_SLOPE_THRESHOLD": 0.0001,
    "PREDICTION_WINDOW": 60,        # 用于预测的历史数据点数
    "PREDICTION_MULTIPLIER": 20,    # 乘数（用于放大斜率的影响）
    "EXPECTED_PROFIT_MULTIPLIER": 1,  # 默认预期收益乘数（动态调整时可能会变）
    "THRESHOLD_SCORE_BUY": 60,      # 做多阀值
    "THRESHOLD_SCORE_SELL": 40,     # 做空阀值
    "MAX_PURCHASES_PER_ROUND": 3    # 每轮最多下单数
}

VERSION = "1.2.5.9.8"
