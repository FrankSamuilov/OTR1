import sys
import time
import math
import warnings
import numpy as np
import tensorflow as tf
from indicators_module import calculate_indicators, score_market, calculate_optimized_indicators, wait_for_entry_timing

import os
import requests  # 确保 requests 已导入
from logger_setup import get_logger  #  确保正确导入
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade_module import get_max_leverage
from concurrent.futures import ThreadPoolExecutor, as_completed
#有好多重复的引入乱乱的:d
#这里是存放主要东西的地方API也要在这里输入
from indicators_module import calculate_indicators, score_market, calculate_optimized_indicators, wait_for_entry_timing
from data_module import get_historical_data, get_spot_balance, get_futures_balance
from config import CONFIG, VERSION
# 开启 Eager Execution（注意 tf.data 部分仍可能运行在图模式下）
tf.config.run_functions_eagerly(True)
import tensorflow as tf
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)
from binance.client import Client
from config import CONFIG, VERSION
from data_module import get_historical_data, get_spot_balance, get_futures_balance
from indicators_module import calculate_indicators, score_market
from model_module import calculate_advanced_score
from position_module import load_positions


from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from model_module import build_tcn_model, calculate_advanced_score
from indicators_module import calculate_optimized_indicators
import numpy as np
import logger_setup  # ✅ 避免循环导入

class USDCTradeBot:
    def __init__(self, api_key: str, api_secret: str, config: dict):
        self.logger = logger_setup.get_logger()  # ✅ 确保 `logger` 正确初始化


import os
import numpy as np
import tensorflow as tf
from binance.client import Client
from config import CONFIG, VERSION
from data_module import get_historical_data, get_spot_balance, get_futures_balance
from indicators_module import calculate_indicators, score_market, calculate_optimized_indicators, wait_for_entry_timing
from model_module import build_tcn_model, calculate_advanced_score
from position_module import load_positions
from logger_setup import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed


class USDCTradeBot:
    def __init__(self, api_key: str, api_secret: str, config: dict):
        print("初始化 USDCTradeBot...")
        self.config = config
        self.client = Client(api_key, api_secret)
        self.logger = get_logger()
        self.trade_cycle = 0
        self.open_positions = []
        self.tcn_model = build_tcn_model((10, 9))

        model_dir = "models"
        model_path = os.path.join(model_dir, "tcn_model.weights.h5")  # 确保文件名正确
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"已创建模型保存目录: {model_dir}")

        if os.path.exists(model_path):
            try:
                self.tcn_model.load_weights(model_path)
                print(f"已加载预训练模型: {model_path}")
            except Exception as e:
                print(f"加载预训练模型失败: {e}")
        else:
            print(f"警告: 未找到 {model_path}，使用随机初始化模型")

        try:
            test_input = np.ones((1, 10, 9), dtype=np.float32)
            test_output = self.tcn_model.predict(test_input, verbose=0)
            print(f"TCN 模型测试输出: {test_output}")
        except Exception as e:
            print(f"TCN 模型测试失败: {e}")

        print(f"初始化完成，TRADE_PAIRS: {self.config['TRADE_PAIRS']}")

    # 其余方法保持不变...
    def concurrent_entry_timing_detection(self, candidates):
        """
        并行检测多个交易对的入场时机
        """
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(candidates), 10)) as executor:  # 限制最大线程数，避免 API 超载
            future_to_candidate = {
                executor.submit(wait_for_entry_timing, self, cand[0], cand[1],
                                self.calculate_dynamic_order_amount(0.01, get_futures_balance(self.client))): cand
                for cand in candidates
            }
            for future in as_completed(future_to_candidate):
                cand = future_to_candidate[future]
                symbol = cand[0]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    print(f"{symbol} 并行检测失败: {e}")
                    results[symbol] = False
        return results

    def get_futures_balance(self):
        try:
            assets = self.client.futures_account_balance()
            for asset in assets:
                if asset["asset"] == "USDC":
                    return float(asset["balance"])
            return 0.0
        except Exception as e:
            print(f"获取期货余额失败: {e}")
            return 0.0

    def get_grok_suggestion(self, symbol, df_ind):
        latest = df_ind.iloc[-1]
        current_data = self.client.futures_symbol_ticker(symbol=symbol)
        current_price = float(current_data['price']) if current_data else None
        predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
        signal = self.generate_trade_signal(df_ind)

        symbol_data = {
            "Symbol": symbol,
            "Current Price": current_price,
            "Predicted Price (60min)": predicted,
            "TCN Signal": signal,
            "ATR": latest['ATR'],
            "MACD": latest['MACD'],
            "ADX": latest['ADX'],
            "VWAP": latest['VWAP'],
            "Momentum": latest['Momentum'],
            "Current Score": score_market(df_ind)
        }
        print("发送给Grok:", symbol_data)
        # 模拟Grok实时建议，您手动输入我的回复
        grok_response = input("Grok建议: ")
        return self.parse_grok_response(grok_response)

    def parse_grok_response(self, response):
        parts = response.split(", ")
        action = parts[0].split()[1]  # "BUY" 或 "SELL"
        time = float(parts[1].split()[1])  # 时间（分钟）
        profit = float(parts[2].split()[2].strip("%")) / 100  # 收益阈值
        return action, time, profit

    def generate_trade_signal(self, df):
        """
        使用 TCN 生成交易信号，增强数据验证和模型检查
        """
        df = calculate_optimized_indicators(df)
        required_cols = ['open', 'high', 'low', 'close', 'VWAP', 'MACD', 'RSI', 'OBV', 'ATR']
        if len(df) < 10 or not all(col in df.columns for col in required_cols):
            print(f"数据不足或缺失列: {len(df)} 条记录，所需列: {required_cols}")
            return None

        df_subset = df[required_cols].iloc[-10:].ffill().fillna(0)
        features = df_subset.values

        # 加载训练时的均值和标准差
        model_dir = "models"
        mean_path = os.path.join(model_dir, "mean_features.npy")
        std_path = os.path.join(model_dir, "std_features.npy")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean_features = np.load(mean_path)
            std_features = np.load(std_path)
        else:
            print(f"警告: 未找到均值或标准差文件，使用当前特征的均值和标准差")
            mean_features = np.mean(features, axis=0)
            std_features = np.std(features, axis=0) + 1e-10

        # 标准化特征
        features = (features - mean_features) / (std_features + 1e-10)

        # 严格检查特征数据
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"特征数据异常: {features}")
            return None

        if not hasattr(self, 'tcn_model') or self.tcn_model is None:
            print("TCN 模型未初始化")
            return None

        try:
            if features.shape != (10, 9):
                print(f"特征形状错误: 期望 (10, 9)，实际 {features.shape}")
                return None

            print(f"输入特征（标准化后）: {features}")
            tcn_signal = self.tcn_model.predict(np.expand_dims(features, axis=0), verbose=0)[0][0]

            if np.isnan(tcn_signal) or np.isinf(tcn_signal):
                print(f"TCN 输出异常: {tcn_signal}, 模型权重路径: {os.path.join('models', 'tcn_model.weights.h5')}")
                return None
        except Exception as e:
            print(f"TCN 预测失败: {e}")
            return None

        vwap_trend = df['close'].iloc[-1] > df['VWAP'].iloc[-1]
        obv_trend = df['OBV'].iloc[-1] > df['OBV'].iloc[-10]
        self.logger.info(f"TCN Signal: {tcn_signal}, VWAP Trend: {vwap_trend}, OBV Trend: {obv_trend}")

        if tcn_signal > 0.2:  # 调整阈值以捕捉更多信号
            signal = "BUY"
        elif tcn_signal < 0.1:
            signal = "SELL"
        else:
            signal = None

        if signal == "BUY" and not (vwap_trend or obv_trend):
            signal = None
        elif signal == "SELL" and (vwap_trend or obv_trend):
            signal = None

        symbolic_name = df.name if hasattr(df, 'name') else '未知'
        print(f"生成信号 {symbolic_name}: {signal} (TCN: {tcn_signal:.3f})")
        return signal


    def auto_convert_stablecoins_to_usdc(self):
        pass

    def auto_transfer_usdc_to_futures(self):
        pass

    def check_all_balances(self) -> tuple:
        spot = get_spot_balance(self.client)
        futures = get_futures_balance(self.client)
        total = spot + futures
        print(f"\n💰 账户余额: 现货 {spot} USDC, 期货 {futures} USDC, 总计 {total} USDC")
        self.logger.info("账户余额查询", extra={"spot": spot, "futures": futures, "total": total})
        return spot, futures

    def get_best_trade_candidates(self):
        """
        计算所有交易对的评分，筛选出最佳交易候选
        """
        best_candidates = []
        for symbol in self.config["TRADE_PAIRS"]:  # ✅ 正确的缩进
            df = get_historical_data(self.client, symbol)  # ✅ 这行代码要缩进
            if df is None or df.empty:
                continue

            df = calculate_indicators(df)
            base_score = score_market(df)
            final_score = base_score  # ✅ 确保代码结构完整
            best_candidates.append((symbol, final_score))

            # 获取最新市场价格
            current_data = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(current_data['price']) if current_data else None
            if current_price is None:
                continue

            # 预测未来价格
            predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
            if predicted is None:
                continue

            risk = abs(current_price - predicted) / current_price  # 计算风险
            candidate_amount = self.calculate_dynamic_order_amount(risk, self.get_futures_balance())
            best_candidates.append((symbol, final_score, candidate_amount))

        best_candidates.sort(key=lambda x: x[1], reverse=True)
        return best_candidates

    def print_current_positions(self):
        if not self.open_positions:
            print("当前无持仓")
        else:
            print("【当前持仓】")
            for pos in self.open_positions:
                print(pos)

    def load_existing_positions(self):
        self.open_positions = load_positions(self.client)
        self.logger.info("加载现有持仓", extra={"open_positions": self.open_positions})

    def record_open_position(self, symbol: str, side: str, entry_price: float, quantity: float):
        for pos in self.open_positions:
            if pos["symbol"] == symbol and pos.get("side", None) == side:
                total_qty = pos["quantity"] + quantity
                new_entry = (pos["entry_price"] * pos["quantity"] + entry_price * quantity) / total_qty
                pos["entry_price"] = new_entry
                pos["quantity"] = total_qty
                pos["max_profit"] = max(pos["max_profit"], 0)
                self.logger.info("合并持仓", extra={"position": pos})
                return
        new_pos = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "open_time": time.time(),
            "quantity": quantity,
            "max_profit": 0.0
        }
        self.open_positions.append(new_pos)
        self.logger.info("记录新持仓", extra={"position": new_pos})

    def manage_open_positions(self):
        print("【持仓管理】")
        self.load_existing_positions()
        for pos in self.open_positions:
            current_data = self.client.futures_symbol_ticker(symbol=pos["symbol"])
            current_price = float(current_data['price']) if current_data else None
            if current_price is None:
                continue
            side = pos.get("side", "BUY" if pos["quantity"] > 0 else "SELL")
            if side.upper() == "BUY":
                actual_profit = (current_price - pos["entry_price"]) * pos["quantity"]
            else:
                actual_profit = (pos["entry_price"] - current_price) * pos["quantity"]
            holding_time = (time.time() - pos["open_time"]) / 60
            print(f"{pos['symbol']} 实际收益: {actual_profit:.2f} USDC, 持仓时长: {holding_time:.1f} 分钟")
            self.logger.info("持仓收益状态", extra={"symbol": pos["symbol"], "actual_profit": actual_profit,
                                                    "holding_time": holding_time})
            if holding_time >= 1440 and actual_profit < 0:
                print(f"{pos['symbol']} 持仓超过24小时且亏损，考虑平仓")
                if holding_time >= 2880:
                    print(f"{pos['symbol']} 超过48小时，强制平仓")
                    self.close_position(pos["symbol"])

    def close_position(self, symbol: str):
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for p in positions:
                amt = float(p.get('positionAmt', 0))
                if abs(amt) > 0:
                    side = p.get("side")
                    if side is None:
                        side = "BUY" if amt < 0 else "SELL"
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=str(abs(amt)),
                        reduceOnly=True
                    )
                    self.logger.info("平仓成功", extra={"symbol": symbol, "order": order})
                    print(f"{symbol} 平仓成功: {order}")
            self.open_positions = [pp for pp in self.open_positions if pp["symbol"] != symbol]
        except Exception as e:
            self.logger.error("平仓失败", extra={"symbol": symbol, "error": str(e)})
            print(f"❌ {symbol} 平仓失败: {e}")

    def display_position_sell_timing(self):
        """
        显示当前持仓的卖出预测，确保使用 **开仓价格（Entry Price）** 而不是 Binance 的 **标记价格（Mark Price）**。
        """
        positions = self.client.futures_position_information()
        if not positions:
            return
        print("【当前持仓卖出预测】")
        self.logger.info("持仓卖出预测")

        for pos in positions:
            amt = float(pos.get('positionAmt', 0))
            entry_price = float(pos.get("entryPrice", 0))  # ✅ 改为获取开仓价格
            symbol = pos["symbol"]

            if abs(amt) > 0:
                current_data = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(current_data['price']) if current_data else None
                predicted = self.predict_short_term_price(symbol, horizon_minutes=60)

                if current_price and predicted:
                    df = get_historical_data(self.client, symbol)
                    if df is None or df.empty:
                        continue

                    slope = self.calculate_slope(df['close'])
                    effective_slope = slope if abs(slope) > self.config["MIN_SLOPE_THRESHOLD"] else self.config[
                        "MIN_SLOPE_THRESHOLD"]

                    # 计算预计平仓时间
                    from trade_module import calculate_expected_time
                    est_time = calculate_expected_time(entry_price, predicted, effective_slope,
                                                       self.config["MIN_SLOPE_THRESHOLD"],
                                                       multiplier=10, max_minutes=150)

                    side = pos.get("side")
                    if side is None:
                        side = "BUY" if amt > 0 else "SELL"

                    # ✅ 计算实际收益（基于开仓价格 entry_price，而非当前价格）
                    if side.upper() == "BUY":
                        profit = (predicted - entry_price) * abs(amt)  # 计算做多收益
                    else:
                        profit = (entry_price - predicted) * abs(amt)  # 计算做空收益

                    print(
                        f"{symbol}: 开仓 {entry_price:.4f}, 预测 {predicted:.4f}, "
                        f"预计需 {est_time:.1f} 分钟, 预期收益 {profit:.2f} USDC")

                    self.logger.info("持仓卖出预测",
                                     extra={"symbol": symbol, "entry_price": entry_price, "predicted": predicted,
                                            "minutes_needed": est_time, "profit": profit})


    def _raw_predict(self, symbol: str) -> float:
        df = get_historical_data(self.client, symbol)
        if df is None or df.empty:
            return None
        close_prices = df['close']
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)
        return slope * len(close_prices)

    def predict_next_price(self, symbol: str) -> float:
        raw_prediction = self._raw_predict(symbol)
        if raw_prediction is None:
            return None
        adjustment = self.adjustment_factors.get(symbol, 0)
        return raw_prediction * (1 + adjustment)

    def predict_short_term_price(self, symbol: str, horizon_minutes: float) -> float:
        df = get_historical_data(self.client, symbol)
        if df is None or df.empty or len(df) < 10:
            print(f"{symbol} 数据不足，无法预测")
            return None
        window_length = self.config.get("PREDICTION_WINDOW", 60)
        window = df['close'].tail(window_length) if len(df) >= window_length else df['close']
        smoothed = window.rolling(window=3, min_periods=1).mean().bfill()  # 替换为 bfill()
        current_price = smoothed.iloc[-1]
        x = np.arange(len(smoothed))
        slope, _ = np.polyfit(x, smoothed, 1)
        multiplier = self.config.get("PREDICTION_MULTIPLIER", 20)
        candles_needed = horizon_minutes / 15.0
        predicted_price = current_price + slope * candles_needed * multiplier

        # 确保预测价格与趋势一致
        if slope > 0 and predicted_price < current_price:
            predicted_price = current_price * 1.01  # 至少上涨 1%
        elif slope < 0 and predicted_price > current_price:
            predicted_price = current_price * 0.99  # 至少下跌 1%

        hist_max = window.max()
        hist_min = window.min()
        buffer = 0.01 * current_price
        predicted_price = min(max(predicted_price, hist_min - buffer), hist_max + buffer)
        return predicted_price

    def record_prediction_error(self, symbol: str):
        current_data = self.client.futures_symbol_ticker(symbol=symbol)
        current = float(current_data['price']) if current_data else None
        raw_pred = self._raw_predict(symbol)
        if current is None or raw_pred is None:
            return
        error = (raw_pred - current) / current
        if symbol not in self.prediction_logs:
            self.prediction_logs[symbol] = []
        self.prediction_logs[symbol].append(error)
        if len(self.prediction_logs[symbol]) >= 20:
            avg_error = sum(self.prediction_logs[symbol]) / len(self.prediction_logs[symbol])
            self.adjustment_factors[symbol] = avg_error
            self.logger.info("更新调整因子", extra={"symbol": symbol, "adjustment_factor": avg_error})
            self.prediction_logs[symbol] = []

    def calculate_slope(self, series):
        x = np.arange(len(series))
        y = series.values
        if len(x) < 2:
            return 0.0
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def get_dynamic_thresholds(self, candidates):
        if not candidates:
            return self.config["THRESHOLD_SCORE_BUY"], self.config["THRESHOLD_SCORE_SELL"], self.config.get(
                "EXPECTED_PROFIT_MULTIPLIER", 1)
        avg_score = sum([cand[1] for cand in candidates]) / len(candidates)
        if avg_score < 50:
            dynamic_buy = self.config["THRESHOLD_SCORE_BUY"] - 5
            dynamic_sell = self.config["THRESHOLD_SCORE_SELL"]
            profit_multiplier = 5
        else:
            dynamic_buy = self.config["THRESHOLD_SCORE_BUY"]
            dynamic_sell = self.config["THRESHOLD_SCORE_SELL"]
            profit_multiplier = 1
        return dynamic_buy, dynamic_sell, profit_multiplier

    def calculate_dynamic_order_amount(self, risk: float, futures: float) -> float:
        if risk < 0.01:
            percentage = 0.20
        elif risk < 0.02:
            percentage = 0.50
        elif risk < 0.03:
            percentage = 0.25
        else:
            percentage = 0.05
        amount = futures * percentage
        if amount < self.config["MIN_NOTIONAL"]:
            amount = self.config["MIN_NOTIONAL"]
        return amount

    def place_futures_order_usdc(self, symbol: str, side: str, amount: float, leverage: int = 20) -> bool:
        """
        使用默认 20 倍杠杆下单，若失败则调整为最大支持杠杆。
        """
        try:
            order_amount = max(amount, self.config["MIN_NOTIONAL"])
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            raw_qty = order_amount / price

            from trade_module import get_precise_quantity, format_quantity, get_max_leverage
            info = self.client.futures_exchange_info()
            step_size = None
            for item in info['symbols']:
                if item['symbol'] == symbol:
                    for f in item['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            break
                    break
            if step_size is None:
                print(f"❌ 未能获取 {symbol} 的步长信息")
                return False
            precision = int(round(-math.log(step_size, 10), 0))

            qty = get_precise_quantity(self.client, symbol, raw_qty)
            qty_str = format_quantity(qty, precision)
            qty = float(qty_str)
            notional = qty * price

            if notional < self.config["MIN_NOTIONAL"]:
                print(
                    f"⚠️ 调整订单数量: {symbol} 计算的金额 {notional:.2f} USDC 小于 {self.config['MIN_NOTIONAL']} USDC")
                desired_qty = self.config["MIN_NOTIONAL"] / price
                qty = get_precise_quantity(self.client, symbol, desired_qty)
                qty_str = format_quantity(qty, precision)
                qty = float(qty_str)
                notional = qty * price
                if notional < self.config["MIN_NOTIONAL"]:
                    msg = f"{symbol} 调整后的金额 {notional:.2f} USDC 仍不足，跳过"
                    print(msg)
                    self.logger.info(msg)
                    return False

            # 尝试设置杠杆
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            except Exception as e:
                err = str(e)
                if "leverage not valid" in err.lower() or "invalid leverage" in err.lower():
                    # 如果 20 倍失败，获取最大杠杆并重试
                    max_leverage = get_max_leverage(self.client, symbol, default_max=20)
                    if max_leverage != leverage:  # 避免重复尝试相同杠杆
                        print(f"⚠️ {symbol} 不支持 {leverage}x 杠杆，调整为 {max_leverage}x")
                        self.client.futures_change_leverage(symbol=symbol, leverage=max_leverage)
                    else:
                        print(f"❌ {symbol} 设置 {leverage}x 杠杆失败，且无更低选项: {e}")
                        self.logger.error("杠杆设置失败", extra={"symbol": symbol, "error": err})
                        return False
                else:
                    print(f"❌ {symbol} 设置杠杆失败: {e}")
                    self.logger.error("杠杆设置失败", extra={"symbol": symbol, "error": err})
                    return False

            # 下单
            pos_side = "LONG" if side.upper() == "BUY" else "SHORT"
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=str(qty),
                positionSide=pos_side
            )
            print(f"✅ {side} {symbol} 成功, 数量={qty}, 杠杆={leverage}")
            self.logger.info("下单成功", extra={"symbol": symbol, "side": side, "quantity": qty, "leverage": leverage,
                                                "order": order})
            return True
        except Exception as e:
            print(f"❌ {symbol} 下单失败: {e}")
            self.logger.error("下单失败", extra={"symbol": symbol, "error": str(e)})
            return False

        except Exception as e:
            err = str(e)
            if "Leverage" in err:
                new_leverage = leverage - 1
                if new_leverage < 1:
                    new_leverage = 1
                print(f"⚠️ 杠杆 {leverage} 不合法，尝试降低至 {new_leverage}")
                self.logger.error("下单失败，降低杠杆重试", extra={"symbol": symbol, "error": err})
                return self.place_futures_order_usdc(symbol, side, amount, new_leverage)
            self.logger.error("下单失败", extra={"symbol": symbol, "error": err})
            print(f"❌ 下单失败: {err}")
            return False

    def add_to_position(self, symbol: str, side: str, amount: float, leverage: int = 3) -> bool:
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            quantity = round(amount / price, 6)
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            pos_side = "LONG" if side.upper() == "BUY" else "SHORT"
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=str(quantity),
                positionSide=pos_side
            )
            print(f"✅ 加单 {side} {symbol} 成功, 数量={quantity}, 杠杆={leverage}")
            self.logger.info("加单成功",
                             extra={"symbol": symbol, "side": side, "quantity": quantity, "leverage": leverage,
                                    "order": order})
            return True
        except Exception as e:
            print(f"❌ 加单失败: {e}")
            self.logger.error("加单失败", extra={"symbol": symbol, "error": str(e)})
            return False

    # 新增：用于根据持仓情况调整候选分数（目前仅返回原始分数）
    def adjust_candidate_score_for_time(self, symbol: str, score: float) -> float:
        return score

    # 新增：用于根据 GARCH 模型调整分数，目前作为占位直接返回传入分数
    def adjust_score_with_garch(self, symbol: str, score: float) -> float:
        return score


    def hedge_and_place_order(self, symbol: str, final_score: float, candidate_amount: float, obs_result: bool) -> bool:
        """
        🚀 交易策略调整：允许双向持仓，但不进行对冲
        """
        if final_score >= self.config["THRESHOLD_SCORE_BUY"]:
            new_side = "BUY"
        elif final_score <= self.config["THRESHOLD_SCORE_SELL"]:
            new_side = "SELL"
        else:
            print(f"{symbol} 分数 {final_score:.2f} 不满足交易条件，不执行交易")
            return False

        # ✅ 允许双向持仓，但不进行对冲
        print(f"{symbol} 交易执行: {new_side}，交易金额 {candidate_amount:.2f} USDC")
        return self.place_futures_order_usdc(symbol, new_side, candidate_amount)



    def update_lstm_online(self, X_train, y_train):
        from lstm_module import online_update_lstm, save_lstm_model
        self.lstm_model = online_update_lstm(self.lstm_model, X_train, y_train, epochs=1, batch_size=32)
        save_lstm_model(self.lstm_model, path="lstm_model.h5")
        self.logger.info("LSTM在线更新完成")

    def trade(self):
        print("进入 trade 方法...")
        while True:
            try:
                self.trade_cycle += 1
                print(f"\n==== 交易轮次 {self.trade_cycle} ====")
                if self.open_positions:
                    print("【持仓管理】")
                    self.manage_open_positions()

                print(f"当前 TRADE_PAIRS: {self.config['TRADE_PAIRS']}")
                if not self.config["TRADE_PAIRS"]:
                    print("⚠️ TRADE_PAIRS 为空，请检查配置")
                    time.sleep(60)
                    continue

                best_candidates = []
                plan_msg = "【本轮详细计划】\n"
                invalid_symbols = []

                for symbol in self.config["TRADE_PAIRS"]:
                    print(f"处理交易对: {symbol}")
                    df = get_historical_data(self.client, symbol)
                    if df is None or df.empty or len(df) < 52:
                        print(
                            f"{symbol} 数据获取失败或不足（长度: {len(df) if df is not None else 'None'}，需要至少52根K线），跳过")
                        continue
                    try:
                        df = calculate_indicators(df)
                        print(
                            f"{symbol} 指标计算完成，MACD: {df['MACD'].iloc[-1] if 'MACD' in df.columns else '未计算'}, "
                            f"OBV: {df['OBV'].iloc[-1] if 'OBV' in df.columns else '未计算'}, "
                            f"ADX: {df['ADX'].iloc[-1] if 'ADX' in df.columns else '未计算'}")
                        base_score = score_market(df)
                        print(f"{symbol} 评分完成，Base Score: {base_score}")
                    except Exception as e:
                        print(f"{symbol} 指标计算或评分失败: {e}")
                        continue

                    adjusted_score = self.adjust_candidate_score_for_time(symbol, base_score)
                    anomaly_score = 0.0
                    adv_score = calculate_advanced_score(df)
                    final_score = adjusted_score + anomaly_score + adv_score
                    final_score = self.adjust_score_with_garch(symbol, final_score)
                    current_data = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(current_data['price']) if current_data else None
                    predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
                    if current_price is None or predicted is None:
                        print(f"{symbol} 价格或预测数据缺失，跳过")
                        continue
                    risk = abs(current_price - predicted) / current_price
                    candidate_amount = self.calculate_dynamic_order_amount(risk, get_futures_balance(self.client))
                    est_qty = candidate_amount / current_price
                    est_profit = (predicted - current_price) * est_qty
                    plan_msg += (f"{symbol}: 基础评分: {base_score:.2f}, 调整后评分: {adjusted_score:.2f}, "
                                 f"高级分: {adv_score:.2f}, 最终评分: {final_score:.2f}\n"
                                 f"当前价格: {current_price:.4f}, 预测价格: {predicted:.4f}, "
                                 f"预期收益: {est_profit:.2f} USDC\n"
                                 f"风险偏差: {risk * 100:.2f}%\n"
                                 f"计划下单金额: {candidate_amount:.2f} USDC\n\n")
                    best_candidates.append((symbol, final_score, candidate_amount))

                print(plan_msg)
                self.logger.info("详细计划", extra={"plan": plan_msg})

                dynamic_buy, dynamic_sell, profit_multiplier = self.get_dynamic_thresholds(best_candidates)
                print(
                    f"动态 BUY 阀值: {dynamic_buy}, 动态 SELL 阀值: {dynamic_sell}, 预期收益乘数: {profit_multiplier}")

                timing_results = self.concurrent_entry_timing_detection(best_candidates)
                invalid_symbols = [symbol for symbol, result in timing_results.items() if not result]

                purchase_count = 0
                for candidate in best_candidates:
                    if purchase_count >= self.config["MAX_PURCHASES_PER_ROUND"]:
                        break
                    symbol, final_score, candidate_amount = candidate
                    if timing_results.get(symbol, False):
                        print(f"尝试 {symbol}, 最终评分 {final_score:.2f}, 交易金额 {candidate_amount:.2f} USDC")
                        if self.hedge_and_place_order(symbol, final_score, candidate_amount, True):
                            purchase_count += 1
                            print(f"{symbol} 下单成功")

                if purchase_count == 0:
                    print("无合适交易机会或下单失败")
                    if invalid_symbols:
                        print(f"无效信号交易对: {', '.join(invalid_symbols)}")
                    self.logger.info("无合适交易机会或下单失败", extra={"invalid_symbols": invalid_symbols})
                self.display_position_sell_timing()
                time.sleep(60)

            except KeyboardInterrupt:
                print("\n⚠️ 交易机器人已被手动终止。")
                self.logger.warning("交易机器人已被手动终止。")
                break
            except Exception as e:
                error_message = str(e)
                self.logger.error("交易异常", extra={"error": error_message})
                print(f"交易异常: {error_message}")
                time.sleep(5)
                continue

def round_to_five(value):
    """确保 value 保持最多 5 位小数"""
    return round(value, 5)


# 测试示例
values = [0.1234567, 0.12345678, 0.123456, 1.0000001, 2.9999999]
rounded_values = [round_to_five(v) for v in values]
print(rounded_values)  # 输出 [0.12346, 0.12346, 0.12346, 1.0, 3.0]


def get_ipv4_address():
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        return response.json().get("ip", "无法获取 IPv4 地址")
    except Exception as e:
        return f"获取 IPv4 失败: {str(e)}"

def calculate_stop_loss(df, entry_price):
    """
    使用 ATR 计算止损点，避免过早触发止损。
    """
    atr = df['ATR'].iloc[-1]
    stop_loss = entry_price - (atr * 2)  # 2 倍 ATR 作为止损
    return stop_loss

def place_trade(symbol, entry_price, side, amount):
    """
    交易执行逻辑：使用 ATR 计算止损，使用 VWAP 过滤假信号
    """
    df = get_historical_data(client, symbol)
    df = calculate_indicators(df)

    # 确保价格在 VWAP 方向一致
    if (side == "BUY" and entry_price < df['VWAP'].iloc[-1]) or (side == "SELL" and entry_price > df['VWAP'].iloc[-1]):
        print(f"🚨 {symbol} 价格未突破 VWAP，跳过交易")
        return False

    stop_loss = calculate_stop_loss(df, entry_price)
    print(f"📊 {symbol} 交易执行：{side} @ {entry_price}, 止损 @ {stop_loss}")

    # 这里调用 Binance API 执行交易（保留原有代码）
    return True


ipv4_address = get_ipv4_address()
print(f"🌍 服务器 IPv4 地址: {ipv4_address}")

import trade_module  # 先导入 trade_module
from binance.client import Client
from trade_module import get_max_leverage  # 仅导入 get_max_leverage

if __name__ == "__main__":
    API_KEY = "JdDbn4SbVDYmtvO6XzFFGtxfVxIzzb2c1Zg0HcJW6PvdOjD0Nxg03sCIUWZQ0W5a"
    API_SECRET = "qnYFpJAVlbVrKibIETeuN3I35YSeDfY2UJow1GxwkxarubdRNsETkg8rpOhqX5eP"
    bot = USDCTradeBot(API_KEY, API_SECRET, CONFIG)
    bot.trade()
#我账户里面暂时还测不出来哪个1111报错我现在哪个杠杆有问题杠杆我写了个自动对接杠杆然后杠杆他自己会一点一点降低杠杆然后来找适配杠杆但是哪个杠杆如果对接到1了就会无线循环我不知道怎么让他不循环
#然后如果在测试的时候报错2019是我账户钱都用了没钱了他就会提示这个！
#报错1111就是哪个精度问题我上网搜了个也不行也chatgpt 我也问了解决不了。。很烦然后我尝试把目光转到是不是请求太多api可能要限制每秒速度我也只是猜想
#我得翻一下哪个chatgpt给我写的杠杆和哪个1111报错的解决方案现在这些代码算法已经非常完善了现在只有这些跟binance api的一堆乱起八糟的bug特别烦人
#尤其是1111！！！和杠杆对接！！！！这个仿佛就是binance他们的bug怎么改都不行哎。。。