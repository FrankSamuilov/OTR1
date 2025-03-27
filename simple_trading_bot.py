import os
import time
import math
import numpy as np
import pandas as pd
import datetime
from binance.client import Client
from config import CONFIG, VERSION
from data_module import get_historical_data
from indicators_module import calculate_optimized_indicators, get_smc_trend_and_duration, find_swing_points, \
    calculate_fibonacci_retracements
from position_module import load_positions, get_total_position_exposure, calculate_order_amount, \
    adjust_position_for_market_change
from logger_setup import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade_module import get_max_leverage, get_precise_quantity, format_quantity
from quality_module import calculate_quality_score, detect_pattern_similarity, adjust_quality_for_similarity


class EnhancedTradingBot:
    def __init__(self, api_key: str, api_secret: str, config: dict):
        print("初始化 EnhancedTradingBot...")
        self.config = config
        self.client = Client(api_key, api_secret)
        self.logger = get_logger()
        self.trade_cycle = 0
        self.open_positions = []  # 存储持仓信息
        self.api_request_delay = 0.5  # API请求延迟以避免限制
        self.historical_data_cache = {}  # 缓存历史数据
        self.quality_score_history = {}  # 存储质量评分历史
        self.similar_patterns_history = {}  # 存储相似模式历史
        self.hedge_mode_enabled = True  # 默认启用双向持仓

        # 创建日志目录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"已创建日志目录: {log_dir}")

        # 尝试启用双向持仓模式
        try:
            position_mode = self.client.futures_get_position_mode()
            if position_mode['dualSidePosition']:
                print("双向持仓模式已启用")
                self.hedge_mode_enabled = True
            else:
                print("尝试启用双向持仓模式...")
                self.client.futures_change_position_mode(dualSidePosition=True)
                print("已启用双向持仓模式")
                self.hedge_mode_enabled = True
        except Exception as e:
            if "code=-4059" in str(e):
                print("双向持仓模式已启用，无需更改")
                self.hedge_mode_enabled = True
            else:
                print(f"⚠️ 启用双向持仓模式失败: {e}")
                self.logger.error("启用双向持仓模式失败", extra={"error": str(e)})
                self.hedge_mode_enabled = False

        print(f"初始化完成，交易对: {self.config['TRADE_PAIRS']}")

    def get_futures_balance(self):
        """获取USDC期货账户余额"""
        try:
            assets = self.client.futures_account_balance()
            for asset in assets:
                if asset["asset"] == "USDC":
                    return float(asset["balance"])
            return 0.0
        except Exception as e:
            self.logger.error(f"获取期货余额失败: {e}")
            return 0.0

    def get_historical_data_with_cache(self, symbol, interval="15m", limit=200, force_refresh=False):
        """获取历史数据，使用缓存减少API调用"""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()

        # 检查缓存是否存在且有效
        if not force_refresh and cache_key in self.historical_data_cache:
            cache_item = self.historical_data_cache[cache_key]
            # 缓存保留10分钟
            if current_time - cache_item['timestamp'] < 600:
                self.logger.info(f"使用缓存数据: {symbol}")
                return cache_item['data']

        # 获取新数据
        try:
            df = get_historical_data(self.client, symbol)
            if df is not None and not df.empty:
                # 缓存数据
                self.historical_data_cache[cache_key] = {
                    'data': df,
                    'timestamp': current_time
                }
                self.logger.info(f"获取并缓存新数据: {symbol}")
                return df
            else:
                self.logger.warning(f"无法获取{symbol}的数据")
                return None
        except Exception as e:
            self.logger.error(f"获取{symbol}历史数据失败: {e}")
            return None

    def generate_trade_signal(self, df, symbol):
        """基于SMC策略生成交易信号"""
        df.name = symbol  # 设置名称以便在日志中引用

        if df is None or len(df) < 20:
            self.logger.warning(f"{symbol}数据不足，无法生成信号")
            return "HOLD", 0

        try:
            # 计算指标
            df = calculate_optimized_indicators(df)
            if df is None or df.empty:
                self.logger.warning(f"{symbol}指标计算失败")
                return "HOLD", 0

            # 计算质量评分
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config, self.logger)

            # 检查模式相似性
            historical_data = []
            for key, cache_item in self.historical_data_cache.items():
                if key.startswith(symbol) and '_15m_' in key:
                    historical_data.append(cache_item['data'])

            similarity_info = detect_pattern_similarity(df, historical_data, window_length=10,
                                                        similarity_threshold=0.8, logger=self.logger)

            # 根据相似性调整质量评分
            if similarity_info['is_similar']:
                adjusted_score = adjust_quality_for_similarity(quality_score, similarity_info)
                self.logger.info(f"{symbol}检测到相似模式", extra={
                    "similarity": similarity_info['max_similarity'],
                    "similar_time": similarity_info['similar_time'],
                    "original_score": quality_score,
                    "adjusted_score": adjusted_score
                })
                quality_score = adjusted_score

                # 记录到相似模式历史
                self.similar_patterns_history[symbol] = similarity_info

                # 在日志中记录相似度
                similarity_pct = round(similarity_info['max_similarity'] * 100, 2)
                similar_time = similarity_info['similar_time']
                if isinstance(similar_time, pd.Timestamp):
                    similar_time = similar_time.strftime('%Y-%m-%d %H:%M')
                self.logger.info(f"{symbol} 与 {similar_time} 相似，相似程度 {similarity_pct}%")

            # 记录质量评分历史
            if symbol not in self.quality_score_history:
                self.quality_score_history[symbol] = []
            self.quality_score_history[symbol].append({
                'time': datetime.datetime.now(),
                'score': quality_score,
                'metrics': metrics
            })

            # 保留最近50个评分记录
            if len(self.quality_score_history[symbol]) > 50:
                self.quality_score_history[symbol] = self.quality_score_history[symbol][-50:]

            # 根据质量评分和指标确定交易信号
            signal = "HOLD"
            # 趋势分析
            trend, duration, trend_info = get_smc_trend_and_duration(df, self.config, self.logger)

            # 获取支撑/阻力位
            swing_highs, swing_lows = find_swing_points(df)
            fib_levels = calculate_fibonacci_retracements(df)

            # 获取当前价格及关键指标
            current_price = df['close'].iloc[-1]
            supertrend_up = False
            supertrend_down = False

            # 计算SMC订单块
            volume_mean = df['volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['volume'].iloc[-1]
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
            has_order_block = (recent_volume > volume_mean * 1.3 and
                               abs(df['close'].iloc[-1] - df['close'].iloc[-2]) < atr)

            # 综合分析生成信号
            if quality_score >= 7.0:  # 高质量评分
                if trend == "UP" and has_order_block:
                    signal = "BUY"
                    self.logger.info(f"{symbol} 高质量上升趋势 + 订单块，建议买入")
                elif quality_score >= 9.0:  # 极高质量
                    self.logger.info(f"{symbol} 极高质量评分 {quality_score:.2f}，建议手动确认加仓")
                    signal = "BUY"  # 超高质量时默认买入
            elif quality_score <= 3.0:  # 低质量评分
                if trend == "DOWN" and has_order_block:
                    signal = "SELL"
                    self.logger.info(f"{symbol} 低质量下降趋势 + 订单块，建议卖出")
            elif quality_score > 3.0 and quality_score < 7.0:  # 中等质量
                if trend == "UP" and has_order_block:
                    if self.is_near_support(current_price, swing_lows, fib_levels):
                        signal = "BUY"
                        self.logger.info(f"{symbol} 中等质量，接近支撑位，建议买入")
                elif trend == "DOWN" and has_order_block:
                    if self.is_near_resistance(current_price, swing_highs, fib_levels):
                        signal = "SELL"
                        self.logger.info(f"{symbol} 中等质量，接近阻力位，建议卖出")

            # 处理波动较大的市场
            high_volatility = False
            if 'ATR' in df.columns:
                atr_mean = df['ATR'].rolling(20).mean().iloc[-1]
                if atr > atr_mean * 1.5:
                    high_volatility = True

            if high_volatility and quality_score > 4.0 and quality_score < 6.0:
                signal = "BOTH"  # 高波动性市场双向建仓
                self.logger.info(f"{symbol} 高波动市场，考虑双向建仓")

            self.logger.info(f"{symbol} 生成信号: {signal}", extra={
                "quality_score": quality_score,
                "trend": trend,
                "duration": duration,
                "has_order_block": has_order_block,
                "near_support": self.is_near_support(current_price, swing_lows, fib_levels),
                "near_resistance": self.is_near_resistance(current_price, swing_highs, fib_levels),
                "high_volatility": high_volatility
            })

            return signal, quality_score

        except Exception as e:
            self.logger.error(f"{symbol}生成信号失败: {e}")
            return "HOLD", 0

    def is_near_support(self, price, swing_lows, fib_levels, threshold=0.01):
        """检查价格是否接近支撑位"""
        # 检查摆动低点
        for low in swing_lows:
            if abs(price - low) / price < threshold:
                return True

        # 检查斐波那契支撑位
        if fib_levels and len(fib_levels) >= 3:
            for level in fib_levels:
                if abs(price - level) / price < threshold:
                    return True

        return False

    def is_near_resistance(self, price, swing_highs, fib_levels, threshold=0.01):
        """检查价格是否接近阻力位"""
        # 检查摆动高点
        for high in swing_highs:
            if abs(price - high) / price < threshold:
                return True

        # 检查斐波那契阻力位
        if fib_levels and len(fib_levels) >= 3:
            for level in fib_levels:
                if abs(price - level) / price < threshold:
                    return True

        return False

    def predict_short_term_price(self, symbol, horizon_minutes=60):
        """预测短期价格走势"""
        df = self.get_historical_data_with_cache(symbol)
        if df is None or df.empty or len(df) < 20:
            self.logger.warning(f"{symbol}数据不足，无法预测价格")
            return None

        try:
            # 计算指标
            df = calculate_optimized_indicators(df)
            if df is None or df.empty:
                return None

            # 使用简单线性回归预测价格
            window_length = min(self.config.get("PREDICTION_WINDOW", 60), len(df))
            window = df['close'].tail(window_length)
            smoothed = window.rolling(window=3, min_periods=1).mean().bfill()

            x = np.arange(len(smoothed))
            slope, intercept = np.polyfit(x, smoothed, 1)

            current_price = smoothed.iloc[-1]
            candles_needed = horizon_minutes / 15.0  # 假设15分钟K线
            multiplier = self.config.get("PREDICTION_MULTIPLIER", 15)

            predicted_price = current_price + slope * candles_needed * multiplier

            # 确保预测有意义
            if slope > 0 and predicted_price < current_price:
                predicted_price = current_price * 1.01  # 至少上涨1%
            elif slope < 0 and predicted_price > current_price:
                predicted_price = current_price * 0.99  # 至少下跌1%

            # 限制在历史范围内
            hist_max = window.max() * 1.05  # 允许5%的超出
            hist_min = window.min() * 0.95  # 允许5%的超出
            predicted_price = min(max(predicted_price, hist_min), hist_max)

            self.logger.info(f"{symbol}价格预测: {predicted_price:.6f}", extra={
                "current_price": current_price,
                "predicted_price": predicted_price,
                "horizon_minutes": horizon_minutes,
                "slope": slope
            })

            return predicted_price
        except Exception as e:
            self.logger.error(f"{symbol}价格预测失败: {e}")
            return None

    def place_hedge_orders(self, symbol, primary_side, quality_score):
        """根据质量评分和信号放置订单，支持双向持仓"""
        account_balance = self.get_futures_balance()

        if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
            self.logger.warning(f"账户余额不足，无法交易: {account_balance} USDC")
            return False

        # 检查当前持仓
        total_exposure, symbol_exposures = get_total_position_exposure(self.open_positions, account_balance)
        symbol_exposure = symbol_exposures.get(symbol, 0)

        # 计算下单金额
        order_amount, order_pct = calculate_order_amount(
            account_balance,
            symbol_exposure,
            max_total_exposure=85,
            max_symbol_exposure=15,
            default_order_pct=5
        )

        if order_amount <= 0:
            self.logger.warning(f"{symbol}下单金额过小或超出限额")
            return False

        # 双向持仓模式
        if primary_side == "BOTH":
            # 质量评分在中间区域时采用双向持仓
            if 4.0 <= quality_score <= 6.0:
                long_amount = order_amount * 0.6  # 60%做多
                short_amount = order_amount * 0.4  # 40%做空

                long_success = self.place_futures_order_usdc(symbol, "BUY", long_amount)
                time.sleep(1)  # 避免API请求过快
                short_success = self.place_futures_order_usdc(symbol, "SELL", short_amount)

                if long_success and short_success:
                    self.logger.info(f"{symbol}双向持仓成功", extra={
                        "long_amount": long_amount,
                        "short_amount": short_amount,
                        "quality_score": quality_score
                    })
                    return True
                else:
                    self.logger.warning(f"{symbol}双向持仓部分失败", extra={
                        "long_success": long_success,
                        "short_success": short_success
                    })
                    return long_success or short_success
            else:
                # 偏向某一方向
                side = "BUY" if quality_score > 5.0 else "SELL"
                return self.place_futures_order_usdc(symbol, side, order_amount)

        elif primary_side in ["BUY", "SELL"]:
            # 根据评分调整杠杆倍数
            leverage = self.calculate_leverage_from_quality(quality_score)
            return self.place_futures_order_usdc(symbol, primary_side, order_amount, leverage)
        else:
            self.logger.warning(f"{symbol}未知交易方向: {primary_side}")
            return False

    def calculate_leverage_from_quality(self, quality_score):
        """根据质量评分计算合适的杠杆水平"""
        if quality_score >= 9.0:
            return 20  # 最高质量，最高杠杆
        elif quality_score >= 8.0:
            return 15
        elif quality_score >= 7.0:
            return 10
        elif quality_score >= 6.0:
            return 8
        elif quality_score >= 5.0:
            return 5
        elif quality_score >= 4.0:
            return 3
        else:
            return 2  # 默认低杠杆

    def place_futures_order_usdc(self, symbol, side, amount, leverage=5):
        """执行期货市场订单"""
        try:
            # 查询当前市场价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            raw_qty = amount / current_price

            # 处理步长精度
            info = self.client.futures_exchange_info()
            step_size = None
            min_qty = None

            for item in info['symbols']:
                if item['symbol'] == symbol:
                    for f in item['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            min_qty = float(f['minQty'])
                            break
                    break

            if step_size is None:
                self.logger.warning(f"无法获取{symbol}的步长信息")
                return False

            precision = int(round(-math.log(step_size, 10), 0))
            qty = get_precise_quantity(self.client, symbol, raw_qty)
            qty_str = format_quantity(qty, precision)
            qty = float(qty_str)

            # 检查最小交易量
            if qty < min_qty:
                self.logger.warning(f"{symbol}数量{qty}小于最小交易量{min_qty}")
                return False

            # 尝试调整杠杆
            try:
                # 获取最大允许杠杆
                max_leverage = get_max_leverage(self.client, symbol, max_allowed=20)
                leverage = min(leverage, max_leverage)

                attempt = 0
                while attempt < 3 and leverage >= 1:
                    try:
                        self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                        break
                    except Exception as e:
                        if "leverage not valid" in str(e).lower():
                            leverage -= 1
                            attempt += 1
                        else:
                            raise e

                if leverage < 1:
                    self.logger.warning(f"{symbol}杠杆调整失败，使用默认杠杆1")
                    leverage = 1
            except Exception as e:
                self.logger.warning(f"设置{symbol}杠杆失败: {e}")
                leverage = 1

            # 执行交易
            if self.hedge_mode_enabled:
                pos_side = "LONG" if side.upper() == "BUY" else "SHORT"
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=str(qty),
                    positionSide=pos_side
                )
            else:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=str(qty)
                )

            # 记录订单信息
            self.logger.info(f"{symbol} {side} 订单执行成功", extra={
                "amount": amount,
                "quantity": qty,
                "leverage": leverage,
                "price": current_price,
                "order_id": order.get("orderId", "unknown")
            })

            # 更新持仓记录
            self.record_open_position(symbol, side, current_price, qty)

            return True
        except Exception as e:
            self.logger.error(f"{symbol} {side} 订单执行失败: {e}")
            return False

    def record_open_position(self, symbol, side, entry_price, quantity):
        """记录新开的持仓"""
        position_side = "LONG" if side.upper() == "BUY" else "SHORT"

        # 检查是否已有同方向持仓
        for i, pos in enumerate(self.open_positions):
            if pos["symbol"] == symbol and pos.get("position_side", None) == position_side:
                # 合并持仓
                total_qty = pos["quantity"] + quantity
                new_entry = (pos["entry_price"] * pos["quantity"] + entry_price * quantity) / total_qty
                self.open_positions[i]["entry_price"] = new_entry
                self.open_positions[i]["quantity"] = total_qty
                self.open_positions[i]["last_update_time"] = time.time()

                self.logger.info(f"更新{symbol} {position_side}持仓", extra={
                    "new_entry_price": new_entry,
                    "total_quantity": total_qty
                })
                return

        # 添加新持仓
        new_pos = {
            "symbol": symbol,
            "side": side,
            "position_side": position_side,
            "entry_price": entry_price,
            "quantity": quantity,
            "open_time": time.time(),
            "last_update_time": time.time(),
            "max_profit": 0.0,
            "dynamic_take_profit": 0.06,  # 默认6%止盈
            "stop_loss": -0.03,  # 默认3%止损
            "position_id": f"{symbol}_{position_side}_{int(time.time())}"
        }

        self.open_positions.append(new_pos)
        self.logger.info(f"新增{symbol} {position_side}持仓", extra=new_pos)

    def close_position(self, symbol, position_side=None):
        """平仓指定货币对的持仓"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            closed_positions = []

            for pos in positions:
                amt = float(pos.get('positionAmt', 0))
                if abs(amt) > 0:
                    current_side = pos.get('positionSide', 'BOTH')

                    # 如果指定了方向，只平仓该方向
                    if position_side is not None and current_side != position_side:
                        continue

                    close_side = "SELL" if amt > 0 else "BUY"

                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=close_side,
                        type="MARKET",
                        quantity=str(abs(amt)),
                        positionSide=current_side,
                        reduceOnly=True
                    )

                    closed_positions.append({
                        "symbol": symbol,
                        "position_side": current_side,
                        "close_side": close_side,
                        "quantity": abs(amt),
                        "order_id": order.get("orderId", "unknown")
                    })

                    self.logger.info(f"{symbol} {current_side}平仓成功", extra={
                        "quantity": abs(amt),
                        "close_side": close_side,
                        "order_id": order.get("orderId", "unknown")
                    })

            # 更新本地持仓记录
            if position_side:
                self.open_positions = [p for p in self.open_positions if
                                       p["symbol"] != symbol or p["position_side"] != position_side]
            else:
                self.open_positions = [p for p in self.open_positions if p["symbol"] != symbol]

            return len(closed_positions) > 0
        except Exception as e:
            self.logger.error(f"{symbol}平仓失败: {e}")
            return False

    def manage_open_positions(self):
        """管理现有持仓，包括止盈止损"""
        self.load_existing_positions()

        if not self.open_positions:
            self.logger.info("当前无持仓")
            return

        current_time = time.time()
        account_balance = self.get_futures_balance()

        # 更新持仓状态并获取动作建议
        updated_positions, actions = adjust_position_for_market_change(self.open_positions, self.client, self.logger)
        self.open_positions = updated_positions

        # 执行止盈止损动作
        for action in actions:
            symbol = action["symbol"]
            side = action["side"]
            position_side = "LONG" if side == "BUY" else "SHORT"
            action_type = action["action"]
            profit_pct = action["profit_pct"]

            if action_type == "take_profit":
                self.logger.info(f"{symbol} {position_side}达到止盈条件, 利润: {profit_pct:.2%}")
                self.close_position(symbol, position_side)
            elif action_type == "stop_loss":
                self.logger.info(f"{symbol} {position_side}达到止损条件, 亏损: {profit_pct:.2%}")
                self.close_position(symbol, position_side)
            elif action_type == "time_stop":
                self.logger.info(f"{symbol} {position_side}持仓时间过长, 执行时间止损")
                self.close_position(symbol, position_side)

        # 检查是否需要加仓
        self.check_add_position(account_balance)

        # 显示持仓状态
        self.display_positions_status()

    def check_add_position(self, account_balance):
        """检查是否有加仓机会"""
        if not self.open_positions:
            return

        # 为每个持仓检查加仓机会
        for pos in self.open_positions:
            symbol = pos["symbol"]
            side = pos["side"]
            position_side = pos["position_side"]
            entry_price = pos["entry_price"]

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except Exception as e:
                self.logger.warning(f"获取{symbol}价格失败: {e}")
                continue

            # 计算当前利润率
            if position_side == "LONG":
                profit_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price

            # 获取最新数据和质量评分
            df = self.get_historical_data_with_cache(symbol, force_refresh=True)
            if df is None:
                continue

            df = calculate_optimized_indicators(df)
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config, self.logger)

            # 检查质量评分历史
            score_increasing = False
            if symbol in self.quality_score_history and len(self.quality_score_history[symbol]) >= 3:
                recent_scores = [item["score"] for item in self.quality_score_history[symbol][-3:]]
                score_increasing = all(recent_scores[i] < recent_scores[i + 1] for i in range(len(recent_scores) - 1))

            # 决定是否加仓
            should_add = False
            add_reason = ""

            if quality_score >= 9.0:
                # 高质量评分自动加仓
                should_add = True
                add_reason = "极高质量评分"
            elif score_increasing and quality_score >= 7.0:
                # 评分持续上升且较高
                should_add = True
                add_reason = "质量评分持续上升"
            elif profit_pct >= 0.05 and quality_score >= 6.0:
                # 已有盈利，评分尚可
                should_add = True
                add_reason = "已有盈利且评分良好"

            # 执行加仓
            if should_add:
                # 计算加仓金额(账户的2%)
                add_amount = account_balance * 0.02

                # 检查同一货币总敞口限制
                total_exposure, symbol_exposures = get_total_position_exposure(self.open_positions, account_balance)
                symbol_exposure = symbol_exposures.get(symbol, 0)

                if symbol_exposure >= 15:
                    self.logger.info(f"{symbol}已达到最大敞口限制，跳过加仓")
                    continue

                self.logger.info(f"{symbol} {position_side}准备加仓", extra={
                    "reason": add_reason,
                    "quality_score": quality_score,
                    "profit_pct": profit_pct,
                    "add_amount": add_amount
                })

                # 执行加仓
                success = self.place_futures_order_usdc(symbol, side, add_amount)
                if success:
                    self.logger.info(f"{symbol} {position_side}加仓成功")
                else:
                    self.logger.warning(f"{symbol} {position_side}加仓失败")

    def display_positions_status(self):
        """显示所有持仓的状态"""
        if not self.open_positions:
            print("当前无持仓")
            return

        print("\n==== 当前持仓状态 ====")
        print(f"{'交易对':<10} {'方向':<6} {'持仓量':<10} {'开仓价':<10} {'当前价':<10} {'利润率':<8} {'持仓时间':<8}")
        print("-" * 70)

        current_time = time.time()

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos["position_side"]
            quantity = pos["quantity"]
            entry_price = pos["entry_price"]
            open_time = pos["open_time"]

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except:
                current_price = 0.0

            # 计算利润率
            if position_side == "LONG":
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                profit_pct = ((entry_price - current_price) / entry_price) * 100

            # 计算持仓时间
            holding_hours = (current_time - open_time) / 3600

            print(
                f"{symbol:<10} {position_side:<6} {quantity:<10.6f} {entry_price:<10.4f} {current_price:<10.4f} {profit_pct:<8.2f}% {holding_hours:<8.2f}h")

        print("-" * 70)

    def load_existing_positions(self):
        """加载现有持仓"""
        self.open_positions = load_positions(self.client, self.logger)

    def display_position_sell_timing(self):
        """显示持仓的预期卖出时机"""
        if not self.open_positions:
            return

        print("\n==== 持仓卖出预测 ====")
        print(f"{'交易对':<10} {'方向':<6} {'当前价':<10} {'预测价':<10} {'预期收益':<10} {'预计时间':<8}")
        print("-" * 70)

        for pos in self.open_positions:
            symbol = pos["symbol"]
            position_side = pos["position_side"]
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except:
                current_price = 0.0

            # 预测未来价格
            predicted_price = self.predict_short_term_price(symbol)
            if predicted_price is None:
                predicted_price = current_price

            # 计算预期收益
            if position_side == "LONG":
                expected_profit = (predicted_price - entry_price) * quantity
            else:  # SHORT
                expected_profit = (entry_price - predicted_price) * quantity

            # 计算预计时间
            df = self.get_historical_data_with_cache(symbol)
            if df is not None and len(df) > 10:
                window = df['close'].tail(10)
                x = np.arange(len(window))
                slope, _ = np.polyfit(x, window, 1)

                if abs(slope) > 0.00001:
                    minutes_needed = abs((predicted_price - current_price) / slope) * 5
                else:
                    minutes_needed = 60
            else:
                minutes_needed = 60

            print(
                f"{symbol:<10} {position_side:<6} {current_price:<10.4f} {predicted_price:<10.4f} {expected_profit:<10.2f} {minutes_needed:<8.0f}分钟")

        print("-" * 70)

    def display_quality_scores(self):
        """显示所有交易对的质量评分"""
        print("\n==== 质量评分排名 ====")
        print(f"{'交易对':<10} {'评分':<6} {'趋势':<8} {'回测':<8} {'相似模式':<12}")
        print("-" * 50)

        scores = []
        for symbol in self.config["TRADE_PAIRS"]:
            df = self.get_historical_data_with_cache(symbol)
            if df is None:
                continue

            df = calculate_optimized_indicators(df)
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config, self.logger)

            trend = metrics.get("trend", "NEUTRAL")

            # 获取相似度信息
            similarity_info = self.similar_patterns_history.get(symbol, {"max_similarity": 0, "is_similar": False})
            similarity_pct = round(similarity_info["max_similarity"] * 100, 1) if similarity_info["is_similar"] else 0

            scores.append((symbol, quality_score, trend, similarity_pct))

        # 按评分排序
        scores.sort(key=lambda x: x[1], reverse=True)

        for symbol, score, trend, similarity_pct in scores:
            backtest = "N/A"  # 回测暂未实现
            print(f"{symbol:<10} {score:<6.2f} {trend:<8} {backtest:<8} {similarity_pct:<12.1f}%")

        print("-" * 50)

    def trade(self):
        """主交易循环"""
        print("启动交易机器人...")
        self.logger.info("交易机器人启动", extra={"version": VERSION})

        while True:
            try:
                self.trade_cycle += 1
                print(f"\n======== 交易循环 #{self.trade_cycle} ========")
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"当前时间: {current_time}")

                # 获取账户余额
                account_balance = self.get_futures_balance()
                print(f"账户余额: {account_balance:.2f} USDC")
                self.logger.info("账户余额", extra={"balance": account_balance})

                if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
                    print(f"⚠️ 账户余额不足，最低要求: {self.config.get('MIN_MARGIN_BALANCE', 10)} USDC")
                    self.logger.warning("账户余额不足", extra={"balance": account_balance,
                                                               "min_required": self.config.get("MIN_MARGIN_BALANCE",
                                                                                               10)})
                    time.sleep(60)
                    continue

                # 管理现有持仓
                self.manage_open_positions()

                # 获取和分析所有交易对
                trade_candidates = []
                for symbol in self.config["TRADE_PAIRS"]:
                    try:
                        df = self.get_historical_data_with_cache(symbol)
                        if df is None:
                            continue

                        signal, quality_score = self.generate_trade_signal(df, symbol)

                        # 获取当前价格
                        try:
                            ticker = self.client.futures_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                        except:
                            self.logger.warning(f"获取{symbol}价格失败")
                            continue

                        # 预测价格
                        predicted_price = self.predict_short_term_price(symbol)

                        # 计算下单金额
                        total_exposure, symbol_exposures = get_total_position_exposure(self.open_positions,
                                                                                       account_balance)
                        symbol_exposure = symbol_exposures.get(symbol, 0)
                        order_amount, _ = calculate_order_amount(account_balance, symbol_exposure)

                        if signal != "HOLD" and order_amount > 0:
                            trade_candidates.append({
                                "symbol": symbol,
                                "signal": signal,
                                "quality_score": quality_score,
                                "current_price": current_price,
                                "predicted_price": predicted_price,
                                "order_amount": order_amount
                            })
                    except Exception as e:
                        self.logger.error(f"处理{symbol}时出错: {e}")

                # 排序候选交易
                trade_candidates.sort(key=lambda x: x["quality_score"], reverse=True)

                # 显示交易计划
                if trade_candidates:
                    print("\n==== 交易计划 ====")
                    print(f"{'交易对':<10} {'信号':<6} {'质量评分':<10} {'当前价':<10} {'预测价':<10}")
                    print("-" * 50)

                    for candidate in trade_candidates:
                        print(
                            f"{candidate['symbol']:<10} {candidate['signal']:<6} {candidate['quality_score']:<10.2f} {candidate['current_price']:<10.4f} {candidate['predicted_price']:<10.4f}")

                    print("-" * 50)
                else:
                    print("本轮无交易候选")

                # 执行交易
                executed_count = 0
                max_trades = min(self.config.get("MAX_PURCHASES_PER_ROUND", 3), len(trade_candidates))

                for i, candidate in enumerate(trade_candidates):
                    if executed_count >= max_trades:
                        break

                    symbol = candidate["symbol"]
                    signal = candidate["signal"]
                    quality_score = candidate["quality_score"]

                    # 高评分智能交易
                    if quality_score >= 7.0 or (signal == "BOTH" and quality_score >= 4.0):
                        print(f"尝试交易 {symbol}, 信号: {signal}, 质量评分: {quality_score:.2f}")

                        if self.place_hedge_orders(symbol, signal, quality_score):
                            executed_count += 1
                            print(f"✅ {symbol} {signal} 交易成功")
                        else:
                            print(f"❌ {symbol} {signal} 交易失败")

                    # 特别提示高质量机会
                    if quality_score >= 9.0:
                        high_quality_msg = f"⭐ {symbol} 质量评分极高 ({quality_score:.2f})，建议手动关注"
                        print(high_quality_msg)
                        self.logger.info(high_quality_msg)

                # 显示质量评分排名
                self.display_quality_scores()

                # 显示持仓卖出预测
                self.display_position_sell_timing()

                # 循环间隔
                sleep_time = 60
                print(f"\n等待 {sleep_time} 秒进入下一轮...")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                self.logger.info("用户中断，程序结束")
                break
            except Exception as e:
                self.logger.error(f"交易循环异常: {e}")
                print(f"错误: {e}")
                time.sleep(30)


if __name__ == "__main__":
    API_KEY = ""
    API_SECRET = ""

    bot = EnhancedTradingBot(API_KEY, API_SECRET, CONFIG)
    bot.trade()