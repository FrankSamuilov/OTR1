from decimal import Decimal, ROUND_DOWN
import math
import numpy as np
from data_module import get_historical_data  # 用于部分计算


def format_quantity(quantity: float, precision: int) -> str:
    """
    格式化数量，保留指定小数位数。
    """
    quantize_str = Decimal("1." + "0" * precision)
    return str(Decimal(quantity).quantize(quantize_str, rounding=ROUND_DOWN))


def calculate_expected_time(current: float, predicted: float, slope: float, min_slope_threshold: float,
                            multiplier: float = 10, max_minutes: float = 150) -> float:
    """
    计算预计达到目标收益所需的时间（分钟），并将预计时间限制在 max_minutes 内。
    """
    effective_slope = slope if abs(slope) > min_slope_threshold else min_slope_threshold
    minutes_needed = abs((predicted - current) / effective_slope) * multiplier
    return min(minutes_needed, max_minutes)


def calculate_expected_profit_percentage(current: float, predicted: float, time_horizon: float,
                                         reference: float = 300) -> float:
    """
    根据持仓时间计算预期收益百分比。
    """
    profit_pct = (predicted - current) / current * 100
    expected_profit_pct = profit_pct * (time_horizon / reference)
    return expected_profit_pct


def get_lot_info(client, symbol: str) -> (float, int):
    """
    从 Binance futures_exchange_info 中查询指定交易对的最小交易数量和步长，并返回 (min_qty, precision)。
    如果查询失败，则返回默认值 (0.001, 3)。
    """
    try:
        info = client.futures_exchange_info()
        for sym in info["symbols"]:
            if sym["symbol"] == symbol:
                for filt in sym["filters"]:
                    if filt["filterType"] == "LOT_SIZE":
                        min_qty = float(filt["minQty"])
                        step_size = float(filt["stepSize"])
                        precision_str = str(step_size).split('.')[-1].rstrip('0')
                        precision = len(precision_str) if precision_str else 0
                        return min_qty, precision
    except Exception:
        pass
    return 0.001, 3


def calculate_expected_profit_percentage(current, predicted, time_horizon, reference=300):
    profit_pct = (predicted - current) / current * 100
    expected_profit_pct = profit_pct * (time_horizon / reference)
    return expected_profit_pct


def check_long_term_growth(client, symbol, lookback_minutes=2400):
    """
    检查某个币种在过去 lookback_minutes 内是否持续增长（简单示例）。
    """
    df = get_historical_data(client, symbol)
    if df is None or df.empty:
        return False
    if len(df) < 160:
        return False
    recent_df = df.tail(160)
    x = np.arange(len(recent_df))
    y = recent_df['close'].values
    slope, _ = np.polyfit(x, y, 1)
    return slope > 0


# 以下为 USDCTradeBot 内部观察期函数的示例，可在 USDCTradeBot 类中调用：
def wait_for_observation_period(self, symbol: str, base_score: float, time_horizon=300) -> bool:
    """
    观察期内采集评分，观察期结束后计算平均评分，与目标评分比较，决定是否达到入场条件。
    """
    target_score = base_score + self.config["OBSERVATION_EXTRA_SCORE"]
    collected_scores = []
    start_time = time.time()
    end_time = start_time + self.config["OBSERVATION_PERIOD"]
    self.logger.info("进入观察期", extra={"symbol": symbol, "target_score": target_score})

    while time.time() < end_time:
        df = get_historical_data(self.client, symbol)
        if df is None or df.empty:
            time.sleep(self.config["OBSERVATION_INTERVAL"])
            continue
        # 假设 USDCTradeBot 中已有 calculate_indicators 和 score_market 方法
        df_ind = self.calculate_indicators(df)
        current_score = self.score_market(df_ind)
        collected_scores.append(current_score)
        time.sleep(self.config["OBSERVATION_INTERVAL"])

    if collected_scores:
        avg_score = sum(collected_scores) / len(collected_scores)
        self.logger.info("观察期结束", extra={"symbol": symbol, "avg_score": avg_score, "target_score": target_score})
        return avg_score >= target_score
    else:
        return False
