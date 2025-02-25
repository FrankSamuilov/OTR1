import sys
import time
import math
import warnings
import numpy as np
import tensorflow as tf

# 开启 Eager Execution（注意 tf.data 部分仍可能运行在图模式下）
tf.config.run_functions_eagerly(True)

from binance.client import Client
from logger_setup import get_logger
from config import CONFIG, VERSION
from data_module import get_historical_data, get_spot_balance, get_futures_balance
from indicators_module import calculate_indicators, score_market
from model_module import calculate_advanced_score
from trade_module import format_quantity, calculate_expected_time
from position_module import load_positions
from lstm_module import load_lstm_model, online_update_lstm, predict_with_lstm

from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class USDCTradeBot:
    def __init__(self, api_key: str, api_secret: str, config: dict):
        self.config = config
        self.client = Client(api_key, api_secret)
        try:
            server_time = self.client.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            offset = local_time - server_time
            print("Time offset set to:", offset)
            self.client.time_offset = offset
            self.client._get_timestamp = lambda: int(time.time() * 1000) - self.client.time_offset - 2000
        except Exception as e:
            print("Failed to set time offset:", e)
        self.logger = get_logger()
        self.open_positions = load_positions(self.client)
        self.trade_cycle = 0

        # LSTM 模型及在线学习
        self.lstm_model = load_lstm_model(path="lstm_model.h5", input_shape=(60, 1))
        self.lstm_buffer_x = []
        self.lstm_buffer_y = []
        self.lstm_update_threshold = self.config.get("LSTM_UPDATE_THRESHOLD", 10)

        self.prediction_logs = {}
        self.adjustment_factors = {}
        self.asset_precision = {}

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
            self.logger.info("持仓收益状态", extra={"symbol": pos["symbol"], "actual_profit": actual_profit, "holding_time": holding_time})
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
        if df is None or df.empty:
            return None
        window_length = self.config.get("PREDICTION_WINDOW", 60)
        if len(df) < window_length:
            window = df['close']
        else:
            window = df['close'].tail(window_length)
        # 平滑数据：3期简单移动平均
        smoothed = window.rolling(window=3, min_periods=1).mean()
        current_price = smoothed.iloc[-1]
        x = np.arange(len(smoothed))
        slope, _ = np.polyfit(x, smoothed, 1)
        multiplier = self.config.get("PREDICTION_MULTIPLIER", 20)
        candles_needed = horizon_minutes / 15.0
        predicted_price = current_price + slope * candles_needed * multiplier

        # 限制预测价格在历史极值附近（加 1% 缓冲）
        hist_max = window.max()
        hist_min = window.min()
        buffer = 0.01 * current_price
        if predicted_price > hist_max:
            predicted_price = hist_max + buffer
        elif predicted_price < hist_min:
            predicted_price = hist_min - buffer
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
            return self.config["THRESHOLD_SCORE_BUY"], self.config["THRESHOLD_SCORE_SELL"], self.config.get("EXPECTED_PROFIT_MULTIPLIER", 1)
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

    def place_futures_order_usdc(self, symbol: str, side: str, amount: float, leverage: int = 3) -> bool:
        try:
            order_amount = max(amount, self.config["MIN_NOTIONAL"])
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            min_qty, precision = 0.001, 3
            raw_qty = order_amount / price
            qty_str_down = format_quantity(raw_qty, precision)
            qty_down = float(qty_str_down)
            notional_down = qty_down * price
            if notional_down >= self.config["MIN_NOTIONAL"]:
                qty_str = qty_str_down
            else:
                step = 10 ** (-precision)
                desired_qty = self.config["MIN_NOTIONAL"] / price
                qty_up = math.ceil(desired_qty / step) * step
                qty_str_up = format_quantity(qty_up, precision)
                qty_up_val = float(qty_str_up)
                notional_up = qty_up_val * price
                if notional_up >= self.config["MIN_NOTIONAL"]:
                    qty_str = qty_str_up
                else:
                    msg = f"{symbol} 下单名义金额 {notional_up:.2f} USDC 仍不足 {self.config['MIN_NOTIONAL']} USDC，跳过"
                    print(msg)
                    self.logger.info(msg)
                    return False
            qty = float(qty_str)
            notional = qty * price
            if notional < self.config["MIN_NOTIONAL"]:
                msg = f"{symbol} 下单名义金额 {notional:.2f} USDC 小于 {self.config['MIN_NOTIONAL']} USDC，跳过"
                print(msg)
                self.logger.info(msg)
                return False

            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            pos_side = "LONG" if side.upper() == "BUY" else "SHORT"
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=str(qty),
                positionSide=pos_side
            )
            print(f"✅ {side} {symbol} 成功, 数量={qty}, 杠杆={leverage}")
            self.logger.info("下单成功", extra={"symbol": symbol, "side": side, "quantity": qty, "leverage": leverage, "order": order})
            return True
        except Exception as e:
            err = str(e)
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
            self.logger.info("加单成功", extra={"symbol": symbol, "side": side, "quantity": quantity, "leverage": leverage, "order": order})
            return True
        except Exception as e:
            print(f"❌ 加单失败: {e}")
            self.logger.error("加单失败", extra={"symbol": symbol, "error": str(e)})
            return False

    def display_position_sell_timing(self):
        positions = self.client.futures_position_information()
        if not positions:
            return
        print("【当前持仓卖出预测】")
        self.logger.info("显示当前持仓卖出预测")
        for pos in positions:
            amt = float(pos.get('positionAmt', 0))
            if abs(amt) > 0:
                symbol = pos['symbol']
                current_data = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(current_data['price']) if current_data else None
                predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
                if current_price and predicted:
                    df = get_historical_data(self.client, symbol)
                    if df is None or df.empty:
                        continue
                    slope = self.calculate_slope(df['close'])
                    effective_slope = slope if abs(slope) > self.config["MIN_SLOPE_THRESHOLD"] else self.config["MIN_SLOPE_THRESHOLD"]
                    est_time = calculate_expected_time(current_price, predicted, effective_slope, self.config["MIN_SLOPE_THRESHOLD"], multiplier=10, max_minutes=150)
                    side = pos.get("side")
                    if side is None:
                        side = "BUY" if amt > 0 else "SELL"
                    if side.upper() == "BUY":
                        profit = (predicted - current_price) * abs(amt)
                    else:
                        profit = (current_price - predicted) * abs(amt)
                    print(f"{symbol}: 当前 {current_price:.4f}, 预测 {predicted:.4f}, 预计需 {est_time:.1f} 分钟, 预期收益 {profit:.2f} USDC")
                    self.logger.info("持仓卖出预测", extra={"symbol": symbol, "current": current_price, "predicted": predicted, "minutes_needed": est_time, "profit": profit})

    def concurrent_observation(self, candidates):
        results = {}
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            future_to_candidate = {executor.submit(self.wait_for_observation_period, cand[0], cand[1]): cand for cand in candidates}
            for future in as_completed(future_to_candidate):
                cand = future_to_candidate[future]
                symbol = cand[0]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception:
                    results[symbol] = False
        return results

    def wait_for_observation_period(self, symbol: str, base_score: float) -> bool:
        if base_score > self.config["THRESHOLD_SCORE_BUY"]:
            target_score = base_score + self.config["OBSERVATION_EXTRA_SCORE"]
        elif base_score < self.config["THRESHOLD_SCORE_SELL"]:
            target_score = base_score - self.config["OBSERVATION_EXTRA_SCORE"]
        else:
            return True

        start_time = time.time()
        while time.time() - start_time < self.config["OBSERVATION_PERIOD"]:
            elapsed = time.time() - start_time
            remaining = self.config["OBSERVATION_PERIOD"] - elapsed
            df = get_historical_data(self.client, symbol)
            if df is None or df.empty:
                time.sleep(self.config["OBSERVATION_INTERVAL"])
                continue
            df_ind = calculate_indicators(df)
            current_score = score_market(df_ind)
            predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
            current_data = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(current_data['price']) if current_data else None
            if current_price is not None and predicted is not None:
                slope = self.calculate_slope(df_ind['close'])
                est_time = calculate_expected_time(current_price, predicted, slope, self.config["MIN_SLOPE_THRESHOLD"], multiplier=10, max_minutes=150)
                expected_profit_pct = (predicted - current_price) / current_price * 100 * (60 / 300)
                print(f"观察中 {symbol}: 当前评分 {current_score:.2f}, 目标评分 {target_score:.2f}, 剩余 {remaining:.0f} 秒, 预计平仓时间 {est_time:.1f} 分钟, 预期收益 {expected_profit_pct:.2f}%")
                self.logger.info("观察期数据", extra={"symbol": symbol, "current_score": current_score, "target_score": target_score, "remaining": remaining, "est_exit_time": est_time, "expected_profit_pct": expected_profit_pct})
            else:
                print(f"观察中 {symbol}: 当前评分 {current_score:.2f}, 目标评分 {target_score:.2f}, 剩余 {remaining:.0f} 秒")
                self.logger.info("观察期数据", extra={"symbol": symbol, "current_score": current_score, "target_score": target_score, "remaining": remaining})
            if (base_score > self.config["THRESHOLD_SCORE_BUY"] and current_score >= target_score) or \
               (base_score < self.config["THRESHOLD_SCORE_SELL"] and current_score <= target_score):
                self.logger.info("观察期内达到目标评分", extra={"symbol": symbol, "current_score": current_score})
                return True
            time.sleep(self.config["OBSERVATION_INTERVAL"])
        return False

    def adjust_candidate_score_for_time(self, symbol: str, score: float) -> float:
        for pos in self.open_positions:
            if pos["symbol"] == symbol:
                holding_time = (time.time() - pos["open_time"]) / 60
                if holding_time >= 600:
                    score -= 10
                elif holding_time >= 300:
                    score -= 5
        return score

    def adjust_score_with_garch(self, symbol: str, score: float) -> float:
        return score

    def update_lstm_online(self, X_train, y_train):
        from lstm_module import online_update_lstm, save_lstm_model
        self.lstm_model = online_update_lstm(self.lstm_model, X_train, y_train, epochs=1, batch_size=32)
        save_lstm_model(self.lstm_model, path="lstm_model.h5")
        self.logger.info("LSTM在线更新完成")

    def online_learning_step(self, symbol: str):
        df = get_historical_data(self.client, symbol)
        if df is None or df.empty or len(df) < 61:
            return
        window = df['close'].values[-61:]
        X = window[:-1].reshape(1, 60, 1)
        y = np.array([window[-1]])
        self.lstm_buffer_x.append(X)
        self.lstm_buffer_y.append(y)
        if len(self.lstm_buffer_x) >= self.lstm_update_threshold:
            X_train = np.vstack(self.lstm_buffer_x)
            y_train = np.concatenate(self.lstm_buffer_y)
            self.update_lstm_online(X_train, y_train)
            self.lstm_buffer_x = []
            self.lstm_buffer_y = []

    def trade(self):
        while True:
            try:
                self.trade_cycle += 1
                print(f"\n==== 交易轮次 {self.trade_cycle} ====")
                self.load_existing_positions()
                self.print_current_positions()
                self.auto_convert_stablecoins_to_usdc()
                self.auto_transfer_usdc_to_futures()
                spot, futures = self.check_all_balances()
                if futures < self.config["MIN_NOTIONAL"]:
                    print("⚠️ 期货账户USDC余额不足，等待资金补充...")
                    time.sleep(300)
                    continue

                if self.open_positions:
                    print("【持仓管理】")
                    self.manage_open_positions()

                for symbol in self.config["TRADE_PAIRS"]:
                    self.online_learning_step(symbol)

                best_candidates = []
                plan_msg = "【本轮详细计划】\n"
                for symbol in self.config["TRADE_PAIRS"]:
                    df = get_historical_data(self.client, symbol)
                    if df is None or df.empty:
                        continue
                    df = calculate_indicators(df)
                    base_score = score_market(df)
                    adjusted_score = self.adjust_candidate_score_for_time(symbol, base_score)
                    anomaly_score = 0.0
                    adv_score = calculate_advanced_score(df)
                    final_score = adjusted_score + anomaly_score + adv_score
                    final_score = self.adjust_score_with_garch(symbol, final_score)
                    current_data = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(current_data['price']) if current_data else None
                    predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
                    if current_price is None or predicted is None:
                        continue
                    risk = abs(current_price - predicted) / current_price
                    candidate_amount = self.calculate_dynamic_order_amount(risk, futures)
                    est_qty = candidate_amount / current_price
                    est_profit = (predicted - current_price) * est_qty
                    plan_msg += (f"{symbol}: 基础评分: {base_score:.2f}, 调整后评分: {adjusted_score:.2f}, "
                                 f"高级分: {adv_score:.2f}, 最终评分: {final_score:.2f}\n"
                                 f"当前价格: {current_price:.4f}, 预测价格: {predicted:.4f}, "
                                 f"预期收益: {est_profit:.2f} USDC\n"
                                 f"风险偏差: {risk*100:.2f}%\n"
                                 f"计划下单金额: {candidate_amount:.2f} USDC\n\n")
                    best_candidates.append((symbol, final_score, candidate_amount))
                print(plan_msg)
                self.logger.info("详细计划", extra={"plan": plan_msg})

                dynamic_buy, dynamic_sell, profit_multiplier = self.get_dynamic_thresholds(best_candidates)
                print(f"动态 BUY 阀值: {dynamic_buy}, 动态 SELL 阀值: {dynamic_sell}, 预期收益乘数: {profit_multiplier}")

                borderline = [(cand[0], cand[1]) for cand in best_candidates]
                obs_results = self.concurrent_observation(borderline)

                best_candidates.sort(key=lambda x: x[1], reverse=True)
                purchase_count = 0
                order_executed = False
                for candidate in best_candidates:
                    if purchase_count >= self.config["MAX_PURCHASES_PER_ROUND"]:
                        break
                    symbol, final_score, candidate_amount = candidate
                    if not obs_results.get(symbol, False):
                        print(f"{symbol} 观察未达标，跳过")
                        continue
                    if final_score > dynamic_buy:
                        side = "BUY"
                    elif final_score < dynamic_sell:
                        side = "SELL"
                    else:
                        continue
                    print(f"尝试 {symbol}, 最终评分 {final_score:.2f}, 交易金额 {candidate_amount:.2f} USDC, 准备下单 {side}")
                    df_test = get_historical_data(self.client, symbol)
                    if df_test is None or df_test.empty:
                        continue
                    new_score = score_market(calculate_indicators(df_test))
                    print(f"刷新后 {symbol} 评分: {new_score:.2f}")
                    leverage = min(int(abs(new_score) / 2 + 1), self.config["MAX_LEVERAGE"])
                    if self.place_futures_order_usdc(symbol, side, candidate_amount, leverage):
                        price_now = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
                        self.record_open_position(symbol, side, price_now, candidate_amount / price_now)
                        order_executed = True
                        purchase_count += 1

                if not order_executed:
                    print("无合适交易机会或下单失败，等待下一轮")
                    self.logger.info("无合适交易机会或下单失败")
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
                if "has no attribute" in error_message:
                    self.logger.error("检测到属性错误，自动退出程序")
                    sys.exit(1)
                time.sleep(5)

def main():
    API_KEY = "你的API_KEY"         # 替换为实际 API_KEY
    API_SECRET = "你的API_SECRET"   # 替换为实际 API_SECRET
    bot = USDCTradeBot(API_KEY, API_SECRET, CONFIG)
    bot.trade()

if __name__ == "__main__":
    main()
