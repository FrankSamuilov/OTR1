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
from pivot_points_module import calculate_pivot_points, analyze_pivot_point_strategy
from advanced_indicators import calculate_smi, calculate_stochastic, calculate_parabolic_sar
from smc_enhanced_prediction import enhanced_smc_prediction, multi_timeframe_smc_prediction
from risk_management import adaptive_risk_management
from integration_module import calculate_enhanced_indicators, comprehensive_market_analysis, generate_trade_recommendation
from logger_utils import Colors, print_colored
import datetime
import time
from integration_module import calculate_enhanced_indicators, generate_trade_recommendation
from multi_timeframe_module import MultiTimeframeCoordinator


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

        # 多时间框架协调器初始化
        self.mtf_coordinator = MultiTimeframeCoordinator(self.client, self.logger)
        print("✅ 多时间框架协调器初始化完成")

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
        """基于SMC策略和多时间框架协调生成交易信号"""
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
            print_colored(f"{symbol} 初始质量评分: {quality_score:.2f}", Colors.INFO)

            # 使用多时间框架协调器
            print_colored(f"🔄 对{symbol}执行多时间框架分析", Colors.BLUE + Colors.BOLD)

            # 获取多时间框架分析的信号
            signal, adjusted_score, details = self.mtf_coordinator.generate_signal(symbol, quality_score)

            # 获取主导时间框架
            primary_tf = details["primary_timeframe"]
            print_colored(f"主导时间框架: {primary_tf}", Colors.INFO)

            # 获取一致性信息
            coherence = details["coherence"]
            print_colored(
                f"时间框架一致性: {coherence['agreement_level']} "
                f"(得分: {coherence['coherence_score']:.1f}/100)",
                Colors.INFO
            )

            # 获取当前价格和预测价格
            try:
                current_data = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(current_data['price']) if current_data else None

                predicted = self.predict_short_term_price(symbol, horizon_minutes=60)

                # 计算预期波动幅度
                price_volatility = 0
                if current_price and predicted:
                    price_volatility = abs(predicted - current_price) / current_price * 100
                    print_colored(f"预测价格波动: {price_volatility:.2f}%", Colors.INFO)
            except Exception as e:
                self.logger.error(f"获取{symbol}价格预测失败: {e}")
                price_volatility = 0
                current_price = None
                predicted = None

            # ===== 提高购买门槛 =====
            # 1. 最小波动幅度要求
            volatility_threshold = 2.0  # 最小波动幅度要求(%)
            if price_volatility < volatility_threshold:
                print_colored(f"❌ {symbol} 预期波动幅度({price_volatility:.2f}%)不足{volatility_threshold}%，不交易",
                              Colors.WARNING)
                return "HOLD", adjusted_score

            # 2. 时间框架一致性要求
            coherence_threshold = 70.0  # 最小一致性评分要求
            if coherence["coherence_score"] < coherence_threshold:
                print_colored(
                    f"❌ {symbol} 时间框架一致性({coherence['coherence_score']:.1f})不足{coherence_threshold}，不交易",
                    Colors.WARNING)
                return "HOLD", adjusted_score

            # 3. 质量评分门槛
            quality_threshold = 7.5  # 高质量评分要求
            if adjusted_score < quality_threshold and "BUY" in signal:
                print_colored(f"❌ {symbol} 质量评分({adjusted_score:.2f})不足{quality_threshold}，不交易",
                              Colors.WARNING)
                return "HOLD", adjusted_score

            # 4. 添加趋势强度门槛
            if 'ADX' in df.columns:
                adx = df['ADX'].iloc[-1]
                if adx < 25:  # ADX低于25表示趋势不明显
                    print_colored(f"❌ {symbol} ADX({adx:.2f})过低，趋势不明显，不交易",
                                  Colors.WARNING)
                    return "HOLD", adjusted_score

            # 记录调整后的质量评分
            print_colored(f"调整后质量评分: {adjusted_score:.2f}", Colors.INFO)

            # 记录信号生成过程到日志
            self.logger.info(f"{symbol} 信号生成", extra={
                "original_score": quality_score,
                "adjusted_score": adjusted_score,
                "primary_timeframe": primary_tf,
                "coherence_level": coherence["agreement_level"],
                "coherence_score": coherence["coherence_score"],
                "dominant_trend": coherence["dominant_trend"],
                "signal": signal,
                "timeframe_conflicts": coherence["trend_conflicts"],
                "price_volatility": price_volatility
            })

            return signal, adjusted_score

        except Exception as e:
            self.logger.error(f"{symbol}生成信号失败: {e}")
            return "HOLD", 0



    def place_hedge_orders(self, symbol, primary_side, quality_score):
        """
        根据质量评分和信号放置订单，支持双向持仓 - 修复版
        """
        account_balance = self.get_futures_balance()

        if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
            self.logger.warning(f"账户余额不足，无法交易: {account_balance} USDC")
            return False

        # 计算下单金额，确保不超过账户余额的5%
        order_amount = account_balance * 0.05
        print(f"📊 账户余额: {account_balance} USDC, 下单金额: {order_amount:.2f} USDC (5%)")

        # 双向持仓模式
        if primary_side == "BOTH":
            # 质量评分在中间区域时采用双向持仓
            if 4.0 <= quality_score <= 6.0:
                # 使用6:4比例分配多空仓位
                long_ratio = 0.6
                short_ratio = 0.4

                long_amount = order_amount * long_ratio
                short_amount = order_amount * short_ratio

                print(f"🔄 执行双向持仓 - 多头: {long_amount:.2f} USDC, 空头: {short_amount:.2f} USDC")

                # 计算每个方向的杠杆
                long_leverage = self.calculate_leverage_from_quality(quality_score)
                short_leverage = max(1, long_leverage - 2)  # 空头杠杆略低

                # 先执行多头订单
                long_success = self.place_futures_order_usdc(symbol, "BUY", long_amount, long_leverage)
                time.sleep(1)
                # 再执行空头订单
                short_success = self.place_futures_order_usdc(symbol, "SELL", short_amount, short_leverage)

                return long_success or short_success
            else:
                # 偏向某一方向
                side = "BUY" if quality_score > 5.0 else "SELL"
                leverage = self.calculate_leverage_from_quality(quality_score)
                return self.place_futures_order_usdc(symbol, side, order_amount, leverage)

        elif primary_side in ["BUY", "SELL"]:
            # 根据评分调整杠杆倍数
            leverage = self.calculate_leverage_from_quality(quality_score)
            return self.place_futures_order_usdc(symbol, primary_side, order_amount, leverage)
        else:
            self.logger.warning(f"{symbol}未知交易方向: {primary_side}")
            return False

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

    def place_futures_order_usdc(self, symbol: str, side: str, amount: float, leverage: int = 5) -> bool:
        """
        执行期货市场订单 - 修复版，解决保证金不足问题
        """
        import math
        import time
        from logger_utils import Colors, print_colored

        try:
            # 获取当前账户余额
            account_balance = self.get_futures_balance()
            print(f"📊 当前账户余额: {account_balance:.2f} USDC")

            # 严格限制订单金额不超过账户余额的5%
            max_allowed_amount = account_balance * 0.05

            if amount > max_allowed_amount:
                print(f"⚠️ 订单金额 {amount:.2f} USDC 超过账户余额5%限制，已调整为 {max_allowed_amount:.2f} USDC")
                amount = max_allowed_amount

            # 确保最低订单金额
            min_amount = self.config.get("MIN_NOTIONAL", 5)
            if amount < min_amount and account_balance >= min_amount:
                amount = min_amount

            # 获取交易对信息
            info = self.client.futures_exchange_info()

            step_size = None
            min_qty = None
            notional_min = None

            # 查找该交易对的所有过滤器
            for item in info['symbols']:
                if item['symbol'] == symbol:
                    for f in item['filters']:
                        # 数量精度
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            min_qty = float(f['minQty'])
                            max_qty = float(f['maxQty'])
                        # 最小订单价值
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            notional_min = float(f.get('notional', 0))
                    break

            # 确保找到了必要的信息
            if step_size is None:
                print_colored(f"❌ {symbol} 无法获取交易精度信息", Colors.ERROR)
                return False

            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 计算数量并应用精度限制
            raw_qty = amount / current_price

            # 计算实际需要的保证金
            margin_required = amount / leverage
            if margin_required > account_balance:
                print(f"❌ 保证金不足: 需要 {margin_required:.2f} USDC, 账户余额 {account_balance:.2f} USDC")
                return False

            # 应用数量精度
            precision = int(round(-math.log(step_size, 10), 0))
            quantity = math.floor(raw_qty * 10 ** precision) / 10 ** precision

            # 确保数量>=最小数量
            if quantity < min_qty:
                print_colored(f"⚠️ {symbol} 数量 {quantity} 小于最小交易量 {min_qty}，已调整", Colors.WARNING)
                quantity = min_qty

            # 格式化为字符串(避免科学计数法问题)
            qty_str = f"{quantity:.{precision}f}"

            # 检查最小订单价值
            notional = quantity * current_price
            if notional_min and notional < notional_min:
                print_colored(f"⚠️ {symbol} 订单价值 ({notional:.2f}) 低于最小要求 ({notional_min})", Colors.WARNING)
                new_qty = math.ceil(notional_min / current_price * 10 ** precision) / 10 ** precision
                quantity = max(min_qty, new_qty)
                qty_str = f"{quantity:.{precision}f}"
                notional = quantity * current_price

            print_colored(f"🔢 {symbol} 计划交易: 金额={amount:.2f} USDC, 数量={quantity}, 价格={current_price}",
                          Colors.INFO)
            print_colored(f"🔢 杠杆: {leverage}倍, 实际保证金: {notional / leverage:.2f} USDC", Colors.INFO)

            # 设置杠杆
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                print(f"✅ {symbol} 设置杠杆成功: {leverage}倍")
            except Exception as e:
                print(f"⚠️ {symbol} 设置杠杆失败: {e}，使用默认杠杆 1")
                leverage = 1

            # 执行交易
            try:
                if hasattr(self, 'hedge_mode_enabled') and self.hedge_mode_enabled:
                    # 双向持仓模式
                    pos_side = "LONG" if side.upper() == "BUY" else "SHORT"
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=qty_str,
                        positionSide=pos_side
                    )
                else:
                    # 单向持仓模式
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=qty_str
                    )

                print_colored(f"✅ {side} {symbol} 成功, 数量={quantity}, 杠杆={leverage}倍", Colors.GREEN)
                self.logger.info(f"{symbol} {side} 订单成功", extra={
                    "order_id": order.get("orderId", "unknown"),
                    "quantity": quantity,
                    "notional": notional,
                    "leverage": leverage
                })

                # 记录持仓信息
                self.record_open_position(symbol, side, current_price, quantity)
                return True

            except Exception as e:
                order_error = str(e)
                print_colored(f"❌ {symbol} {side} 订单执行失败: {order_error}", Colors.ERROR)

                if "insufficient balance" in order_error.lower() or "margin is insufficient" in order_error.lower():
                    print_colored(f"  原因: 账户余额或保证金不足", Colors.WARNING)
                    print_colored(f"  当前余额: {account_balance} USDC, 需要保证金: {notional / leverage:.2f} USDC",
                                  Colors.WARNING)
                elif "precision" in order_error.lower():
                    print_colored(f"  原因: 价格或数量精度不正确", Colors.WARNING)
                elif "lot size" in order_error.lower():
                    print_colored(f"  原因: 订单大小不符合要求", Colors.WARNING)
                elif "min notional" in order_error.lower():
                    print_colored(f"  原因: 订单价值低于最小要求", Colors.WARNING)

                self.logger.error(f"{symbol} {side} 交易失败", extra={"error": order_error})
                return False

        except Exception as e:
            print_colored(f"❌ {symbol} {side} 交易过程中发生错误: {e}", Colors.ERROR)
            self.logger.error(f"{symbol} 交易错误", extra={"error": str(e)})
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
        """
        平仓指定货币对的持仓，增强版本 - 修复平仓失败问题

        参数:
            symbol: 交易对符号
            position_side: 持仓方向 ('LONG', 'SHORT', None=全部平仓)

        返回:
            success: 是否成功平仓
            closed_positions: 已平仓的持仓信息列表
        """
        try:
            print(f"🔄 正在尝试平仓 {symbol} {position_side if position_side else '全部持仓'}")

            # 获取当前持仓信息
            positions = self.client.futures_position_information(symbol=symbol)
            if not positions:
                print(f"⚠️ 未找到 {symbol} 的持仓信息")
                return False, []

            # 筛选有实际持仓量的记录
            active_positions = [pos for pos in positions if abs(float(pos.get('positionAmt', 0))) > 0]
            if not active_positions:
                print(f"⚠️ {symbol} 没有活跃持仓")
                return False, []

            print(f"📊 {symbol} 找到 {len(active_positions)} 个活跃持仓")

            # 跟踪已平仓的持仓
            closed_positions = []
            success = False

            for pos in active_positions:
                amt = float(pos.get('positionAmt', 0))
                current_side = pos.get('positionSide', 'BOTH')

                # 如果指定了方向，只平仓该方向
                if position_side is not None and current_side != position_side:
                    print(f"➡️ 跳过 {symbol} {current_side} 持仓 (不匹配请求的方向 {position_side})")
                    continue

                # 确定平仓方向
                close_side = "SELL" if amt > 0 else "BUY"

                # 格式化数量，确保精度正确
                quantity = abs(amt)

                # 获取交易所数量精度信息
                info = self.client.futures_exchange_info()
                step_size = None

                for item in info['symbols']:
                    if item['symbol'] == symbol:
                        for f in item['filters']:
                            if f['filterType'] == 'LOT_SIZE':
                                step_size = float(f['stepSize'])
                                break
                        break

                # 应用精度
                if step_size:
                    precision = 0
                    while step_size < 1:
                        step_size *= 10
                        precision += 1

                    quantity_str = f"{quantity:.{precision}f}"
                else:
                    # 默认精度
                    quantity_str = f"{quantity:.6f}"

                print(f"🔄 执行平仓: {symbol} {current_side}, 方向: {close_side}, 数量: {quantity_str}")

                try:
                    # 创建市价平仓订单
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=close_side,
                        type="MARKET",
                        quantity=quantity_str,
                        positionSide=current_side,
                        reduceOnly=True
                    )

                    print(f"✅ {symbol} {current_side} 平仓成功! 订单ID: {order.get('orderId', 'unknown')}")

                    # 记录平仓信息
                    closed_positions.append({
                        "symbol": symbol,
                        "position_side": current_side,
                        "close_side": close_side,
                        "quantity": quantity,
                        "order_id": order.get("orderId", "unknown")
                    })

                    success = True

                    # 记录日志
                    self.logger.info(f"{symbol} {current_side} 平仓成功", extra={
                        "quantity": quantity,
                        "close_side": close_side,
                        "order_id": order.get("orderId", "unknown")
                    })

                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ {symbol} {current_side} 平仓失败: {error_msg}")

                    # 记录详细错误
                    if "insufficient balance" in error_msg.lower():
                        print(f"  原因: 账户余额不足")
                    elif "lot size" in error_msg.lower():
                        print(f"  原因: 订单大小不符合要求, 尝试调整精度")
                    elif "precision" in error_msg.lower():
                        print(f"  原因: 数量精度不正确")

                    self.logger.error(f"{symbol} {current_side} 平仓失败", extra={"error": error_msg})

                    # 尝试使用替代方法平仓 - 使用position_information中的精确数量
                    try:
                        print(f"🔄 尝试替代方法平仓: {symbol} {current_side}")

                        # 重新获取持仓信息
                        updated_pos = self.client.futures_position_information(symbol=symbol)
                        matching_pos = [p for p in updated_pos if
                                        p.get('positionSide') == current_side and float(p.get('positionAmt', 0)) != 0]

                        if matching_pos:
                            # 使用系统提供的精确数量
                            precise_amt = matching_pos[0]['positionAmt']

                            # 创建市价平仓订单，不转换数量格式
                            order = self.client.futures_create_order(
                                symbol=symbol,
                                side=close_side,
                                type="MARKET",
                                quantity=str(abs(float(precise_amt))),
                                positionSide=current_side,
                                reduceOnly=True
                            )

                            print(f"✅ 替代方法平仓成功! 订单ID: {order.get('orderId', 'unknown')}")
                            success = True

                            # 记录平仓信息
                            closed_positions.append({
                                "symbol": symbol,
                                "position_side": current_side,
                                "close_side": close_side,
                                "quantity": abs(float(precise_amt)),
                                "order_id": order.get("orderId", "unknown")
                            })

                            self.logger.info(f"{symbol} {current_side} 替代方法平仓成功", extra={
                                "quantity": abs(float(precise_amt)),
                                "order_id": order.get("orderId", "unknown")
                            })
                        else:
                            print(f"⚠️ 找不到匹配的持仓进行替代平仓")
                    except Exception as alt_e:
                        print(f"❌ 替代平仓方法也失败: {alt_e}")
                        self.logger.error(f"{symbol} {current_side} 替代平仓失败", extra={"error": str(alt_e)})

            # 更新本地持仓记录
            if success:
                if position_side:
                    self.open_positions = [p for p in self.open_positions if
                                           p["symbol"] != symbol or p.get("position_side") != position_side]
                else:
                    self.open_positions = [p for p in self.open_positions if p["symbol"] != symbol]

                print(f"✅ 成功平仓 {len(closed_positions)} 个 {symbol} 持仓")

            return success, closed_positions

        except Exception as e:
            print(f"❌ {symbol} 平仓过程中发生错误: {e}")
            self.logger.error(f"{symbol} 平仓过程发生错误", extra={"error": str(e)})
            return False, []

    def manage_open_positions(self):
        """管理现有持仓，包括止盈止损 - 修复版"""
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
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 止盈平仓成功!")
                else:
                    print(f"❌ {symbol} {position_side} 止盈平仓失败")
            elif action_type == "stop_loss":
                self.logger.info(f"{symbol} {position_side}达到止损条件, 亏损: {profit_pct:.2%}")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 止损平仓成功!")
                else:
                    print(f"❌ {symbol} {position_side} 止损平仓失败")
            elif action_type == "time_stop":
                self.logger.info(f"{symbol} {position_side}持仓时间过长, 执行时间止损")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 时间止损平仓成功!")
                else:
                    print(f"❌ {symbol} {position_side} 时间止损平仓失败")

        for pos in self.open_positions:
            symbol = pos["symbol"]
            side = pos.get("side", "BUY")
            position_side = pos.get("position_side", "LONG")
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]
            holding_time = (current_time - pos["open_time"]) / 3600  # 小时

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except Exception as e:
                print(f"⚠️ 无法获取 {symbol} 当前价格: {e}")
                continue

            # 计算盈亏
            if position_side == "LONG" or side == "BUY":
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price

            # 获取止盈止损设置
            take_profit = pos.get("dynamic_take_profit", 0.06)
            stop_loss = pos.get("stop_loss", -0.03)

            profit_color = Colors.GREEN if profit_pct >= 0 else Colors.RED
            print(
                f"{symbol} {position_side}: 持仓 {holding_time:.2f}小时, 当前盈亏 {profit_color}{profit_pct:.2%}{Colors.RESET}, "
                f"止盈线 {take_profit:.2%}, 止损线 {stop_loss:.2%}"
            )

            # 检查是否应该触发止盈止损，但尚未被自动触发
            if profit_pct >= take_profit:
                print(f"⚠️ {symbol} {position_side} 已达止盈条件，正在手动平仓...")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 手动止盈平仓成功!")
                else:
                    print(f"❌ {symbol} {position_side} 手动止盈平仓失败")
            elif profit_pct <= stop_loss:
                print(f"⚠️ {symbol} {position_side} 已达止损条件，正在手动平仓...")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 手动止损平仓成功!")
                else:
                    print(f"❌ {symbol} {position_side} 手动止损平仓失败")
            # 检查持仓时间是否过长 (超过24小时)
            elif holding_time > 24 and profit_pct < 0:
                print(f"⚠️ {symbol} {position_side} 持仓时间过长且处于亏损状态，正在手动平仓...")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 手动时间止损平仓成功!")
                else:
                    print(f"❌ {symbol} {position_side} 手动时间止损平仓失败")

        # 重新加载持仓以确保数据最新
        self.load_existing_positions()

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
        """主交易循环 - 集成多时间框架分析"""
        print("启动多时间框架集成交易机器人...")
        self.logger.info("多时间框架集成交易机器人启动", extra={"version": "MTF-" + VERSION})

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

                # 分析所有交易对并生成建议
                trade_candidates = []
                for symbol in self.config["TRADE_PAIRS"]:
                    try:
                        print(f"\n分析交易对: {symbol}")
                        # 获取历史数据
                        df = self.get_historical_data_with_cache(symbol, force_refresh=True)
                        if df is None:
                            print(f"❌ 无法获取{symbol}数据")
                            continue

                        # 使用新的信号生成函数
                        signal, quality_score = self.generate_trade_signal(df, symbol)

                        # 跳过保持信号
                        if signal == "HOLD":
                            print(f"⏸️ {symbol} 保持观望")
                            continue

                        # 获取当前价格
                        try:
                            ticker = self.client.futures_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                        except Exception as e:
                            print(f"❌ 获取{symbol}价格失败: {e}")
                            continue

                        # 预测未来价格
                        predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
                        if predicted is None:
                            predicted = current_price * (1.05 if signal == "BUY" else 0.95)  # 默认5%变动

                        # 计算风险和交易金额
                        risk = abs(current_price - predicted) / current_price

                        # 处理轻量级信号
                        if signal.startswith("LIGHT_"):
                            actual_signal = signal.replace("LIGHT_", "")
                            candidate_amount = self.calculate_dynamic_order_amount(risk, account_balance) * 0.5  # 半仓
                            print_colored(f"{symbol} 轻仓位{actual_signal}信号，使用50%标准仓位", Colors.YELLOW)
                        else:
                            actual_signal = signal
                            candidate_amount = self.calculate_dynamic_order_amount(risk, account_balance)

                        # 添加到候选列表
                        candidate = {
                            "symbol": symbol,
                            "signal": actual_signal,
                            "quality_score": quality_score,
                            "current_price": current_price,
                            "predicted_price": predicted,
                            "risk": risk,
                            "amount": candidate_amount,
                            "is_light": signal.startswith("LIGHT_")
                        }

                        trade_candidates.append(candidate)

                        print_colored(
                            f"候选交易: {symbol} {actual_signal}, "
                            f"质量评分: {quality_score:.2f}, "
                            f"预期波动: {risk * 100:.2f}%, "
                            f"下单金额: {candidate_amount:.2f} USDC",
                            Colors.GREEN if actual_signal == "BUY" else Colors.RED
                        )

                    except Exception as e:
                        self.logger.error(f"处理{symbol}时出错: {e}")
                        print(f"❌ 处理{symbol}时出错: {e}")

                # 按质量评分排序候选交易
                trade_candidates.sort(key=lambda x: x["quality_score"], reverse=True)

                # 显示详细交易计划
                if trade_candidates:
                    print("\n==== 详细交易计划 ====")
                    for idx, candidate in enumerate(trade_candidates, 1):
                        symbol = candidate["symbol"]
                        signal = candidate["signal"]
                        quality = candidate["quality_score"]
                        current = candidate["current_price"]
                        predicted = candidate["predicted_price"]
                        amount = candidate["amount"]
                        is_light = candidate["is_light"]

                        side_color = Colors.GREEN if signal == "BUY" else Colors.RED
                        position_type = "轻仓位" if is_light else "标准仓位"

                        print(f"\n{idx}. {symbol} - {side_color}{signal}{Colors.RESET} ({position_type})")
                        print(f"   质量评分: {quality:.2f}")
                        print(f"   当前价格: {current:.6f}, 预测价格: {predicted:.6f}")
                        print(f"   预期波动: {candidate['risk'] * 100:.2f}%")
                        print(f"   下单金额: {amount:.2f} USDC")
                else:
                    print("\n本轮无交易候选")

                # 执行交易
                executed_count = 0
                max_trades = min(self.config.get("MAX_PURCHASES_PER_ROUND", 3), len(trade_candidates))

                for candidate in trade_candidates:
                    if executed_count >= max_trades:
                        break

                    symbol = candidate["symbol"]
                    signal = candidate["signal"]
                    amount = candidate["amount"]
                    quality_score = candidate["quality_score"]

                    print(f"\n🚀 执行交易: {symbol} {signal}, 金额: {amount:.2f} USDC")

                    # 计算适合的杠杆水平
                    leverage = self.calculate_leverage_from_quality(quality_score)

                    # 执行交易
                    if self.place_futures_order_usdc(symbol, signal, amount, leverage):
                        executed_count += 1
                        print(f"✅ {symbol} {signal} 交易成功")
                    else:
                        print(f"❌ {symbol} {signal} 交易失败")

                # 显示持仓卖出预测
                self.display_position_sell_timing()

                # 打印交易循环总结
                print(f"\n==== 交易循环总结 ====")
                print(f"分析交易对: {len(self.config['TRADE_PAIRS'])}个")
                print(f"交易候选: {len(trade_candidates)}个")
                print(f"执行交易: {executed_count}个")

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
    API_KEY = "lnfs30CvqF8cCIdRcIfW6kKnGGpLoRzTUrwdRslTX4e7a0O6OJ3SYsUT6gF1B26"
    API_SECRET = "llSlxBLrrxh21ugMzli5x6NveNrwQyLBI7YEgTR4VOMyTmVP6V9uqmrN90hX10c"

    bot = EnhancedTradingBot(API_KEY, API_SECRET, CONFIG)
    bot.trade()