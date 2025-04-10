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

# 导入集成模块（这是最简单的方法，因为它整合了所有其他模块的功能）
from integration_module import (
    calculate_enhanced_indicators,
    comprehensive_market_analysis,
    generate_trade_recommendation
)
import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 在文件开头导入所需的模块后，添加这个类定义
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

    def active_position_monitor(self, check_interval=15):
        """
        主动监控持仓，确保及时执行止盈止损

        参数:
            check_interval: 检查间隔（秒）
        """
        print(f"🔄 启动主动持仓监控（每{check_interval}秒检查一次）")

        try:
            while True:
                # 如果没有持仓，等待一段时间后再检查
                if not self.open_positions:
                    time.sleep(check_interval)
                    continue

                # 当前持仓列表的副本，用于检查
                positions = self.open_positions.copy()

                for pos in positions:
                    symbol = pos["symbol"]
                    position_side = pos.get("position_side", "LONG")
                    entry_price = pos["entry_price"]

                    # 获取当前价格
                    try:
                        ticker = self.client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])
                    except Exception as e:
                        continue

                    # 计算利润百分比
                    if position_side == "LONG":
                        profit_pct = (current_price - entry_price) / entry_price
                    else:  # SHORT
                        profit_pct = (entry_price - current_price) / entry_price

                    # 使用固定的止盈止损比例
                    take_profit = 0.025  # 固定2.5%止盈
                    stop_loss = -0.0175  # 固定1.75%止损

                    # 检查止盈条件
                    if profit_pct >= take_profit:
                        print(
                            f"🔔 主动监控: {symbol} {position_side} 达到止盈条件 ({profit_pct:.2%} >= {take_profit:.2%})")
                        success, closed = self.close_position(symbol, position_side)
                        if success:
                            print(f"✅ {symbol} {position_side} 止盈平仓成功: +{profit_pct:.2%}")
                            self.logger.info(f"{symbol} {position_side}主动监控止盈平仓", extra={
                                "profit_pct": profit_pct,
                                "take_profit": take_profit,
                                "entry_price": entry_price,
                                "exit_price": current_price
                            })
                        else:
                            print(f"❌ {symbol} {position_side} 止盈平仓失败")

                    # 检查止损条件
                    elif profit_pct <= stop_loss:
                        print(
                            f"🔔 主动监控: {symbol} {position_side} 达到止损条件 ({profit_pct:.2%} <= {stop_loss:.2%})")
                        success, closed = self.close_position(symbol, position_side)
                        if success:
                            print(f"✅ {symbol} {position_side} 止损平仓成功: {profit_pct:.2%}")
                            self.logger.info(f"{symbol} {position_side}主动监控止损平仓", extra={
                                "profit_pct": profit_pct,
                                "stop_loss": stop_loss,
                                "entry_price": entry_price,
                                "exit_price": current_price
                            })
                        else:
                            print(f"❌ {symbol} {position_side} 止损平仓失败")

                # 等待下一次检查
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print("主动持仓监控已停止")
        except Exception as e:
            print(f"主动持仓监控发生错误: {e}")
            self.logger.error(f"主动持仓监控错误", extra={"error": str(e)})

    def calculate_dynamic_order_amount(self, risk, account_balance):
        """基于风险和账户余额计算适当的订单金额"""
        # 基础订单百分比 - 默认账户的5%
        base_pct = 5.0

        # 根据风险调整订单百分比
        if risk > 0.05:  # 高风险
            adjusted_pct = base_pct * 0.6  # 减小到基础的60%
        elif risk > 0.03:  # 中等风险
            adjusted_pct = base_pct * 0.8  # 减小到基础的80%
        elif risk < 0.01:  # 低风险
            adjusted_pct = base_pct * 1.2  # 增加到基础的120%
        else:
            adjusted_pct = base_pct

        # 计算订单金额
        order_amount = account_balance * (adjusted_pct / 100)

        # 确保订单金额在合理范围内
        min_amount = 5.0  # 最小5 USDC
        max_amount = account_balance * 0.1  # 最大为账户10%

        order_amount = max(min_amount, min(order_amount, max_amount))

        print_colored(f"动态订单金额: {order_amount:.2f} USDC ({adjusted_pct:.1f}% 账户余额)", Colors.INFO)

        return order_amount

    def trade(self):
        """增强版多时框架集成交易循环，包含主动持仓监控"""
        import threading

        print("启动增强版多时间框架集成交易机器人...")
        self.logger.info("增强版多时间框架集成交易机器人启动", extra={"version": "Enhanced-MTF-" + VERSION})

        # 在单独的线程中启动主动持仓监控
        monitor_thread = threading.Thread(target=self.active_position_monitor, args=(15,), daemon=True)
        monitor_thread.start()
        print("✅ 主动持仓监控已在后台启动（每15秒检查一次）")

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

                        # 检查原始信号是否为轻量级
                        is_light = False
                        # 临时获取原始信号
                        _, _, details = self.mtf_coordinator.generate_signal(symbol, quality_score)
                        raw_signal = details.get("coherence", {}).get("recommendation", "")
                        if raw_signal.startswith("LIGHT_"):
                            is_light = True
                            print_colored(f"{symbol} 检测到轻量级信号，将使用较小仓位", Colors.YELLOW)

                        # 获取当前价格
                        try:
                            ticker = self.client.futures_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                        except Exception as e:
                            print(f"❌ 获取{symbol}价格失败: {e}")
                            continue

                        # 预测未来价格
                        predicted = None
                        if "price_prediction" in details and details["price_prediction"].get("valid", False):
                            predicted = details["price_prediction"]["predicted_price"]
                        else:
                            predicted = self.predict_short_term_price(symbol, horizon_minutes=90)  # 使用90分钟预测

                        if predicted is None:
                            predicted = current_price * (1.05 if signal == "BUY" else 0.95)  # 默认5%变动

                        # 计算预期价格变动百分比
                        expected_movement = abs(predicted - current_price) / current_price * 100

                        # 如果预期变动小于2.5%，则跳过交易
                        if expected_movement < 2.5:
                            print_colored(
                                f"⚠️ {symbol}的预期价格变动({expected_movement:.2f}%)小于最低要求(2.5%)，跳过交易",
                                Colors.WARNING)
                            continue

                        # 计算风险和交易金额
                        risk = expected_movement / 100  # 预期变动作为风险指标

                        # 计算交易金额时考虑轻量级信号
                        candidate_amount = self.calculate_dynamic_order_amount(risk, account_balance)
                        if is_light:
                            candidate_amount *= 0.5  # 轻量级信号使用半仓
                            print_colored(f"{symbol} 轻量级信号，使用50%标准仓位: {candidate_amount:.2f} USDC",
                                          Colors.YELLOW)

                        # 添加到候选列表
                        candidate = {
                            "symbol": symbol,
                            "signal": signal,
                            "quality_score": quality_score,
                            "current_price": current_price,
                            "predicted_price": predicted,
                            "risk": risk,
                            "amount": candidate_amount,
                            "is_light": is_light,
                            "expected_movement": expected_movement
                        }

                        trade_candidates.append(candidate)

                        print_colored(
                            f"候选交易: {symbol} {signal}, "
                            f"质量评分: {quality_score:.2f}, "
                            f"预期波动: {expected_movement:.2f}%, "
                            f"下单金额: {candidate_amount:.2f} USDC",
                            Colors.GREEN if signal == "BUY" else Colors.RED
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
                        expected_movement = candidate["expected_movement"]

                        side_color = Colors.GREEN if signal == "BUY" else Colors.RED
                        position_type = "轻仓位" if is_light else "标准仓位"

                        print(f"\n{idx}. {symbol} - {side_color}{signal}{Colors.RESET} ({position_type})")
                        print(f"   质量评分: {quality:.2f}")
                        print(f"   当前价格: {current:.6f}, 预测价格: {predicted:.6f}")
                        print(f"   预期波动: {expected_movement:.2f}%")
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
                    is_light = candidate["is_light"]

                    print(f"\n🚀 执行交易: {symbol} {signal}, 金额: {amount:.2f} USDC{' (轻仓位)' if is_light else ''}")

                    # 计算适合的杠杆水平
                    leverage = self.calculate_leverage_from_quality(quality_score)
                    if is_light:
                        # 轻仓位降低杠杆
                        leverage = max(1, int(leverage * 0.7))
                        print_colored(f"轻仓位降低杠杆至 {leverage}倍", Colors.YELLOW)

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

    def generate_trade_signal(self, df, symbol):
        """生成更积极的交易信号，降低了预期变动和质量评分阈值"""

        if df is None or len(df) < 20:
            return "HOLD", 0

        try:
            # 计算指标
            df = calculate_optimized_indicators(df)
            if df is None or df.empty:
                return "HOLD", 0

            # 计算质量评分
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config, self.logger)
            print_colored(f"{symbol} 初始质量评分: {quality_score:.2f}", Colors.INFO)

            # 获取多时间框架信号
            signal, adjusted_score, details = self.mtf_coordinator.generate_signal(symbol, quality_score)
            print_colored(f"多时间框架信号: {signal}, 调整后评分: {adjusted_score:.2f}", Colors.INFO)

            # 打印一致性分析详情
            coherence = details.get("coherence", {})
            print_colored(f"{symbol} 一致性分析:", Colors.INFO)
            print_colored(f"  一致性级别: {coherence.get('agreement_level', '未知')}", Colors.INFO)
            print_colored(f"  主导趋势: {coherence.get('dominant_trend', '未知')}", Colors.INFO)
            print_colored(f"  推荐: {coherence.get('recommendation', '未知')}", Colors.INFO)

            # 获取当前价格
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
            except Exception as e:
                return "HOLD", 0

            # 获取价格预测
            predicted_price = self.predict_short_term_price(symbol, horizon_minutes=60)
            if predicted_price is None:
                # 默认假设5%变动
                predicted_price = current_price * (1.05 if signal == "BUY" else 0.95)

            # 计算预期变动
            expected_movement = abs(predicted_price - current_price) / current_price * 100
            print_colored(f"{symbol} 预期价格变动: {expected_movement:.2f}%", Colors.INFO)

            # 降低最小预期变动要求 (从2.5%改为1.0%)
            min_movement = 1.0

            # 只有当信号明确为"NEUTRAL"且预期变动很小时才保持观望
            if signal == "NEUTRAL" and expected_movement < min_movement:
                print_colored(f"{symbol} 无明确信号且预期变动({expected_movement:.2f}%)小于{min_movement}%",
                              Colors.YELLOW)
                return "HOLD", 0

            # 更积极的信号生成 - 降低质量评分阈值
            if adjusted_score >= 5.0 and "BUY" in signal:
                final_signal = "BUY"
            elif adjusted_score <= 5.0 and "SELL" in signal:
                final_signal = "SELL"
            elif coherence.get("recommendation") == "BUY" and adjusted_score >= 4.5:
                final_signal = "BUY"
            elif coherence.get("recommendation") == "SELL" and adjusted_score <= 5.5:
                final_signal = "SELL"
            # 特殊处理黄金ETF
            elif symbol == "PAXGUSDT":
                if adjusted_score >= 5.0:
                    final_signal = "BUY"
                    print_colored(f"为 PAXGUSDT 生成特殊 BUY 信号", Colors.GREEN)
                else:
                    final_signal = "SELL"
                    print_colored(f"为 PAXGUSDT 生成特殊 SELL 信号", Colors.RED)
            else:
                final_signal = "HOLD"

            print_colored(f"{symbol} 最终信号: {final_signal}, 评分: {adjusted_score:.2f}", Colors.INFO)
            return final_signal, adjusted_score

        except Exception as e:
            self.logger.error(f"{symbol} 信号生成失败: {e}")
            return "HOLD", 0

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
        执行期货市场订单 - 增强版，加入预期价格变动检查和固定止盈止损

        参数:
            symbol: 交易对符号
            side: 交易方向 ('BUY' 或 'SELL')
            amount: 交易金额(USDC)
            leverage: 杠杆倍数

        返回:
            bool: 交易是否成功
        """
        import math
        import time
        from logger_utils import Colors, print_colored

        try:
            # 获取当前账户余额
            account_balance = self.get_futures_balance()
            print(f"📊 当前账户余额: {account_balance:.2f} USDC")

            # 获取当前价格
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 预测未来价格，用于检查最小价格变动
            predicted_price = self.predict_short_term_price(symbol, horizon_minutes=60)
            if predicted_price is None:
                predicted_price = current_price * (1.05 if side == "BUY" else 0.95)  # 默认5%变动

            # 计算预期价格变动百分比
            expected_movement = abs(predicted_price - current_price) / current_price * 100

            # 如果预期变动小于2.5%，则跳过交易
            if expected_movement < 2.5:
                print_colored(f"⚠️ {symbol}的预期价格变动({expected_movement:.2f}%)小于最低要求(2.5%)", Colors.WARNING)
                self.logger.warning(f"{symbol}预期变动不足", extra={"expected_movement": expected_movement})
                return False

            # 严格限制订单金额不超过账户余额的5%
            max_allowed_amount = account_balance * 0.05

            if amount > max_allowed_amount:
                print(f"⚠️ 订单金额 {amount:.2f} USDC 超过账户余额5%限制，已调整为 {max_allowed_amount:.2f} USDC")
                amount = max_allowed_amount

            # 确保最低订单金额
            min_amount = self.config.get("MIN_NOTIONAL", 5)
            if amount < min_amount and account_balance >= min_amount:
                amount = min_amount
                print(f"⚠️ 订单金额已调整至最低限额: {min_amount} USDC")

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
            print_colored(f"📈 预期价格变动: {expected_movement:.2f}%, 从 {current_price:.6f} 到 {predicted_price:.6f}",
                          Colors.INFO)

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
                    "leverage": leverage,
                    "expected_movement": expected_movement
                })

                # 记录持仓信息 - 使用固定止盈止损比例
                self.record_open_position(symbol, side, current_price, quantity,
                                          take_profit=0.025,  # 固定2.5%止盈
                                          stop_loss=-0.0175)  # 固定1.75%止损
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

    def record_open_position(self, symbol, side, entry_price, quantity, take_profit=0.025, stop_loss=-0.0175):
        """记录新开的持仓，使用固定的止盈止损比例

        参数:
            symbol: 交易对符号
            side: 交易方向 ('BUY' 或 'SELL')
            entry_price: 入场价格
            quantity: 交易数量
            take_profit: 止盈百分比，默认2.5%
            stop_loss: 止损百分比，默认-1.75%
        """
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

                # 使用固定的止盈止损比例
                self.open_positions[i]["dynamic_take_profit"] = take_profit  # 固定2.5%止盈
                self.open_positions[i]["stop_loss"] = stop_loss  # 固定1.75%止损

                self.logger.info(f"更新{symbol} {position_side}持仓", extra={
                    "new_entry_price": new_entry,
                    "total_quantity": total_qty,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss
                })
                return

        # 添加新持仓，使用固定的止盈止损比例
        new_pos = {
            "symbol": symbol,
            "side": side,
            "position_side": position_side,
            "entry_price": entry_price,
            "quantity": quantity,
            "open_time": time.time(),
            "last_update_time": time.time(),
            "max_profit": 0.0,
            "dynamic_take_profit": take_profit,  # 固定2.5%止盈
            "stop_loss": stop_loss,  # 固定1.75%止损
            "position_id": f"{symbol}_{position_side}_{int(time.time())}"
        }

        self.open_positions.append(new_pos)
        self.logger.info(f"新增{symbol} {position_side}持仓", extra={
            **new_pos,
            "take_profit": take_profit,
            "stop_loss": stop_loss
        })

        print_colored(
            f"📝 新增{symbol} {position_side}持仓，止盈: {take_profit * 100:.2f}%，止损: {abs(stop_loss) * 100:.2f}%",
            Colors.GREEN + Colors.BOLD)

    def close_position(self, symbol, position_side=None):
        """平仓指定货币对的持仓，并记录历史

        参数:
            symbol: 交易对符号
            position_side: 持仓方向 ('LONG' 或 'SHORT')，不指定则平仓所有方向

        返回:
            (success, closed_positions): 平仓是否成功及平仓的持仓列表
        """
        closed_positions = []
        success = False  # 初始化 success 变量

        try:
            # 首先检查本地持仓信息
            positions = []
            for pos in self.open_positions:
                if pos["symbol"] == symbol:
                    if position_side is None or pos.get("position_side", "LONG") == position_side:
                        positions.append(pos)

            if not positions:
                print(f"⚠️ 未找到 {symbol} {position_side or '任意方向'} 的持仓")
                self.logger.warning(f"未找到持仓", extra={"symbol": symbol, "position_side": position_side})
                return False, []

            # 平仓每个匹配的持仓
            for pos in positions:
                side = "SELL" if pos.get("position_side", "LONG") == "LONG" else "BUY"
                quantity = pos["quantity"]
                close_success = False

                print(f"📉 平仓 {symbol} {pos.get('position_side', 'LONG')}, 数量: {quantity}")

                try:
                    # 处理数量格式化
                    try:
                        # 优先使用format_quantity函数如果存在
                        if hasattr(self, 'format_quantity'):
                            formatted_qty = self.format_quantity(symbol, quantity)
                        else:
                            # 备用格式化方法
                            precision = 3  # 默认精度
                            formatted_qty = str(round(float(quantity), precision))
                            # 确保不使用科学计数法
                            if 'e' in formatted_qty.lower():
                                formatted_qty = f"{float(quantity):.8f}".rstrip('0').rstrip('.')
                    except Exception as e:
                        print(f"⚠️ 数量格式化失败: {e}, 尝试直接使用原始数量")
                        formatted_qty = str(quantity)  # 直接使用原始数量的字符串形式

                    print(f"📊 使用格式化数量: {formatted_qty}")

                    # 使用市价单平仓
                    if hasattr(self, 'hedge_mode_enabled') and self.hedge_mode_enabled:
                        # 双向持仓模式
                        order = self.client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type="MARKET",
                            quantity=formatted_qty,
                            positionSide=pos.get("position_side", "LONG")
                        )
                    else:
                        # 单向持仓模式
                        order = self.client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type="MARKET",
                            quantity=formatted_qty,
                            reduceOnly=True
                        )

                    # 平仓成功
                    close_success = True
                    closed_positions.append(pos)

                    # 记录平仓信息
                    self.logger.info(f"{symbol} {pos.get('position_side', 'LONG')} 平仓成功", extra={
                        "quantity": quantity,
                        "exit_side": side,
                        "order_id": order.get("orderId", "unknown")
                    })

                    # 计算持仓时间
                    entry_time = pos.get("open_time", time.time() - 3600)
                    holding_hours = (time.time() - entry_time) / 3600

                    # 获取当前价格作为平仓价格
                    try:
                        ticker = self.client.futures_symbol_ticker(symbol=symbol)
                        exit_price = float(ticker['price'])
                    except Exception as e:
                        print(f"⚠️ 获取退出价格失败: {e}")
                        exit_price = pos.get("entry_price", 0)  # 默认值

                    # 计算盈亏
                    entry_price = pos.get("entry_price", 0)
                    if pos.get("position_side", "LONG") == "LONG":
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        profit_pct = (entry_price - exit_price) / entry_price * 100

                    # 记录完整的持仓历史
                    if hasattr(self, 'position_history'):
                        history_record = {
                            "symbol": symbol,
                            "position_side": pos.get("position_side", "LONG"),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "quantity": quantity,
                            "open_time": entry_time,
                            "close_time": time.time(),
                            "holding_time": holding_hours,
                            "profit_pct": profit_pct,
                            "take_profit": pos.get("dynamic_take_profit", 0.025),
                            "stop_loss": pos.get("stop_loss", -0.0175),
                            "close_reason": "take_profit" if profit_pct > 0 else "stop_loss"
                        }

                        # 添加到历史记录
                        self.position_history.append(history_record)
                        print(f"📝 记录交易历史: {symbol} {pos.get('position_side', 'LONG')} 盈亏: {profit_pct:.2f}%")

                        # 保存到文件
                        if hasattr(self, '_save_position_history'):
                            self._save_position_history()

                except Exception as e:
                    print(f"❌ {symbol} {pos.get('position_side', 'LONG')} 平仓失败: {e}")
                    self.logger.error(f"{symbol} 平仓失败", extra={"error": str(e)})
                    close_success = False

            # 如果有任何一个持仓平仓成功，就认为整体成功
            success = any(pos in closed_positions for pos in positions)

            # 从本地持仓列表中移除已平仓的持仓
            for pos in closed_positions:
                if pos in self.open_positions:
                    self.open_positions.remove(pos)

            # 重新加载持仓以确保数据最新
            self.load_existing_positions()

            return success, closed_positions

        except Exception as e:
            print(f"❌ 平仓过程中发生错误: {e}")
            self.logger.error(f"平仓过程错误", extra={"symbol": symbol, "error": str(e)})
            return False, []

    def manage_open_positions(self):
        """管理现有持仓，确保使用固定的止盈止损比例"""
        self.load_existing_positions()

        if not self.open_positions:
            self.logger.info("当前无持仓")
            return

        current_time = time.time()
        account_balance = self.get_futures_balance()

        # 更新持仓状态 - 固定止盈止损比例
        updated_positions = []

        for pos in self.open_positions:
            # 为所有持仓应用固定的止盈止损比例
            pos["dynamic_take_profit"] = 0.025  # 固定2.5%止盈
            pos["stop_loss"] = -0.0175  # 固定1.75%止损
            updated_positions.append(pos)

        self.open_positions = updated_positions

        # 检查每个持仓的止盈止损条件
        positions_to_remove = []  # 记录需要移除的持仓

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

            # 获取固定的止盈止损比例
            take_profit = pos.get("dynamic_take_profit", 0.025)  # 2.5%
            stop_loss = pos.get("stop_loss", -0.0175)  # -1.75%

            profit_color = Colors.GREEN if profit_pct >= 0 else Colors.RED
            print(
                f"{symbol} {position_side}: 持仓 {holding_time:.2f}小时, 当前盈亏 {profit_color}{profit_pct:.2%}{Colors.RESET}, "
                f"止盈线 {take_profit:.2%}, 止损线 {stop_loss:.2%}"
            )

            # 检查是否达到止盈条件
            if profit_pct >= take_profit:
                print(f"🔔 {symbol} {position_side} 达到止盈条件 ({profit_pct:.2%} >= {take_profit:.2%})，执行平仓...")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 止盈平仓成功!")
                    positions_to_remove.append(pos)
                    self.logger.info(f"{symbol} {position_side}止盈平仓", extra={
                        "profit_pct": profit_pct,
                        "take_profit": take_profit,
                        "entry_price": entry_price,
                        "exit_price": current_price
                    })
                else:
                    print(f"❌ {symbol} {position_side} 止盈平仓失败")

            # 检查是否达到止损条件
            elif profit_pct <= stop_loss:
                print(f"🔔 {symbol} {position_side} 达到止损条件 ({profit_pct:.2%} <= {stop_loss:.2%})，执行平仓...")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 止损平仓成功!")
                    positions_to_remove.append(pos)
                    self.logger.info(f"{symbol} {position_side}止损平仓", extra={
                        "profit_pct": profit_pct,
                        "stop_loss": stop_loss,
                        "entry_price": entry_price,
                        "exit_price": current_price
                    })
                else:
                    print(f"❌ {symbol} {position_side} 止损平仓失败")

            # 检查持仓时间是否过长 (超过24小时)
            elif holding_time > 24:
                print(f"🔔 {symbol} {position_side} 持仓时间过长 ({holding_time:.2f}小时 > 24小时)，执行平仓...")
                success, closed = self.close_position(symbol, position_side)
                if success:
                    print(f"✅ {symbol} {position_side} 时间止损平仓成功!")
                    positions_to_remove.append(pos)
                    self.logger.info(f"{symbol} {position_side}时间止损平仓", extra={
                        "holding_time": holding_time,
                        "profit_pct": profit_pct,
                        "entry_price": entry_price,
                        "exit_price": current_price
                    })
                else:
                    print(f"❌ {symbol} {position_side} 时间止损平仓失败")

        # 从持仓列表中移除已平仓的持仓
        for pos in positions_to_remove:
            if pos in self.open_positions:
                self.open_positions.remove(pos)

        # 重新加载持仓以确保数据最新
        self.load_existing_positions()

        # 显示持仓状态
        self.display_positions_status()

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
            quality_score, metrics = calculate_quality_score(df, self.client, symbol, None, self.config,
                                                             self.logger)

            trend = metrics.get("trend", "NEUTRAL")

            # 获取相似度信息
            similarity_info = self.similar_patterns_history.get(symbol, {"max_similarity": 0, "is_similar": False})
            similarity_pct = round(similarity_info["max_similarity"] * 100, 1) if similarity_info[
                "is_similar"] else 0

            scores.append((symbol, quality_score, trend, similarity_pct))

        # 按评分排序
        scores.sort(key=lambda x: x[1], reverse=True)

        for symbol, score, trend, similarity_pct in scores:
            backtest = "N/A"  # 回测暂未实现
            print(f"{symbol:<10} {score:<6.2f} {trend:<8} {backtest:<8} {similarity_pct:<12.1f}%")

        print("-" * 50)


def _save_position_history(self):
    """保存持仓历史到文件"""
    try:
        with open("position_history.json", "w") as f:
            json.dump(self.position_history, f, indent=4)
    except Exception as e:
        print(f"❌ 保存持仓历史失败: {e}")


def _load_position_history(self):
    """从文件加载持仓历史"""
    try:
        if os.path.exists("position_history.json"):
            with open("position_history.json", "r") as f:
                self.position_history = json.load(f)
        else:
            self.position_history = []
    except Exception as e:
        print(f"❌ 加载持仓历史失败: {e}")
        self.position_history = []


def analyze_position_statistics(self):
    """分析并显示持仓统计数据"""
    # 基本统计
    stats = {
        "total_trades": len(self.position_history),
        "winning_trades": 0,
        "losing_trades": 0,
        "total_profit": 0.0,
        "total_loss": 0.0,
        "avg_holding_time": 0.0,
        "symbols": {},
        "hourly_distribution": [0] * 24,  # 24小时
        "daily_distribution": [0] * 7,  # 周一到周日
    }

    holding_times = []

    for pos in self.position_history:
        profit = pos.get("profit_pct", 0)
        symbol = pos.get("symbol", "unknown")
        holding_time = pos.get("holding_time", 0)  # 小时

        # 按交易对统计
        if symbol not in stats["symbols"]:
            stats["symbols"][symbol] = {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "profit": 0.0,
                "loss": 0.0
            }

        stats["symbols"][symbol]["total"] += 1

        # 胜率与盈亏统计
        if profit > 0:
            stats["winning_trades"] += 1
            stats["total_profit"] += profit
            stats["symbols"][symbol]["wins"] += 1
            stats["symbols"][symbol]["profit"] += profit
        else:
            stats["losing_trades"] += 1
            stats["total_loss"] += abs(profit)
            stats["symbols"][symbol]["losses"] += 1
            stats["symbols"][symbol]["loss"] += abs(profit)

        # 时间统计
        if holding_time > 0:
            holding_times.append(holding_time)

        # 小时分布
        if "open_time" in pos:
            open_time = datetime.datetime.fromtimestamp(pos["open_time"])
            stats["hourly_distribution"][open_time.hour] += 1
            stats["daily_distribution"][open_time.weekday()] += 1

    # 计算平均持仓时间
    if holding_times:
        stats["avg_holding_time"] = sum(holding_times) / len(holding_times)

    # 计算胜率
    if stats["total_trades"] > 0:
        stats["win_rate"] = stats["winning_trades"] / stats["total_trades"] * 100
    else:
        stats["win_rate"] = 0

    # 计算盈亏比
    if stats["total_loss"] > 0:
        stats["profit_loss_ratio"] = stats["total_profit"] / stats["total_loss"]
    else:
        stats["profit_loss_ratio"] = float('inf')  # 无亏损

    # 计算每个交易对的胜率和平均盈亏
    for symbol, data in stats["symbols"].items():
        if data["total"] > 0:
            data["win_rate"] = data["wins"] / data["total"] * 100
            data["avg_profit"] = data["profit"] / data["wins"] if data["wins"] > 0 else 0
            data["avg_loss"] = data["loss"] / data["losses"] if data["losses"] > 0 else 0
            data["net_profit"] = data["profit"] - data["loss"]

    return stats


def generate_statistics_charts(self, stats):
    """生成统计图表"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter

    # 确保目录存在
    charts_dir = "statistics_charts"
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)

    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用兼容的样式

    # 1. 交易对胜率对比图
    plt.figure(figsize=(12, 6))
    symbols = list(stats["symbols"].keys())
    win_rates = [data["win_rate"] for data in stats["symbols"].values()]
    trades = [data["total"] for data in stats["symbols"].values()]

    # 按交易次数排序
    sorted_idx = sorted(range(len(trades)), key=lambda i: trades[i], reverse=True)
    symbols = [symbols[i] for i in sorted_idx]
    win_rates = [win_rates[i] for i in sorted_idx]
    trades = [trades[i] for i in sorted_idx]

    colors = ['green' if wr >= 50 else 'red' for wr in win_rates]

    if symbols:  # 确保有数据
        plt.bar(symbols, win_rates, color=colors)
        plt.axhline(y=50, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('交易对')
        plt.ylabel('胜率 (%)')
        plt.title('各交易对胜率对比')
        plt.xticks(rotation=45)

        # 添加交易次数标签
        for i, v in enumerate(win_rates):
            plt.text(i, v + 2, f"{trades[i]}次", ha='center')

        plt.tight_layout()
        plt.savefig(f"{charts_dir}/symbol_win_rates.png")
    plt.close()

    # 2. 日内交易分布
    plt.figure(figsize=(12, 6))
    plt.bar(range(24), stats["hourly_distribution"])
    plt.xlabel('小时')
    plt.ylabel('交易次数')
    plt.title('日内交易时间分布')
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/hourly_distribution.png")
    plt.close()

    # 3. 每周交易分布
    plt.figure(figsize=(10, 6))
    days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    plt.bar(days, stats["daily_distribution"])
    plt.xlabel('星期')
    plt.ylabel('交易次数')
    plt.title('每周交易日分布')
    plt.tight_layout()
    plt.savefig(f"{charts_dir}/daily_distribution.png")
    plt.close()

    # 4. 交易对净利润对比
    plt.figure(figsize=(12, 6))
    sorted_symbols = sorted(stats["symbols"].items(), key=lambda x: x[1]["total"], reverse=True)
    net_profits = [data["net_profit"] for _, data in sorted_symbols]
    symbols_sorted = [s for s, _ in sorted_symbols]

    if symbols_sorted:  # 确保有数据
        colors = ['green' if np >= 0 else 'red' for np in net_profits]
        plt.bar(symbols_sorted, net_profits, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('交易对')
        plt.ylabel('净利润 (%)')
        plt.title('各交易对净利润对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.savefig(f"{charts_dir}/symbol_net_profits.png")
    plt.close()

    # 5. 盈亏分布图
    if self.position_history:
        profits = [pos.get("profit_pct", 0) for pos in self.position_history]
        plt.figure(figsize=(12, 6))
        sns.histplot(profits, bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('盈亏百分比 (%)')
        plt.ylabel('次数')
        plt.title('交易盈亏分布')
        plt.tight_layout()
        plt.savefig(f"{charts_dir}/profit_distribution.png")
    plt.close()


def generate_statistics_report(self, stats):
    """生成HTML统计报告"""
    report_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>交易统计报告 - {report_time}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .stat-card {{ background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .green {{ color: green; }}
            .red {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .chart {{ width: 48%; margin-bottom: 20px; }}
            @media (max-width: 768px) {{ .chart {{ width: 100%; }} }}
        </style>
    </head>
    <body>
        <h1>交易统计报告</h1>
        <p>生成时间: {report_time}</p>

        <div class="stat-card">
            <h2>总体概览</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>总交易次数</td><td>{stats['total_trades']}</td></tr>
                <tr><td>盈利交易</td><td>{stats['winning_trades']} ({stats['win_rate']:.2f}%)</td></tr>
                <tr><td>亏损交易</td><td>{stats['losing_trades']}</td></tr>
                <tr><td>总盈利</td><td class="green">{stats['total_profit']:.2f}%</td></tr>
                <tr><td>总亏损</td><td class="red">{stats['total_loss']:.2f}%</td></tr>
                <tr><td>净盈亏</td><td class="{('green' if stats['total_profit'] > stats['total_loss'] else 'red')}">{stats['total_profit'] - stats['total_loss']:.2f}%</td></tr>
                <tr><td>盈亏比</td><td>{stats['profit_loss_ratio']:.2f}</td></tr>
                <tr><td>平均持仓时间</td><td>{stats['avg_holding_time']:.2f} 小时</td></tr>
            </table>
        </div>

        <div class="stat-card">
            <h2>交易对分析</h2>
            <table>
                <tr>
                    <th>交易对</th>
                    <th>交易次数</th>
                    <th>胜率</th>
                    <th>平均盈利</th>
                    <th>平均亏损</th>
                    <th>净盈亏</th>
                </tr>
    """

    # 按交易次数排序
    sorted_symbols = sorted(stats["symbols"].items(), key=lambda x: x[1]["total"], reverse=True)

    for symbol, data in sorted_symbols:
        html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>{data['total']}</td>
                    <td>{data['win_rate']:.2f}%</td>
                    <td class="green">{data['avg_profit']:.2f}%</td>
                    <td class="red">{data['avg_loss']:.2f}%</td>
                    <td class="{('green' if data['net_profit'] >= 0 else 'red')}">{data['net_profit']:.2f}%</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="chart-container">
            <div class="chart">
                <h3>交易对胜率对比</h3>
                <img src="statistics_charts/symbol_win_rates.png" width="100%">
            </div>
            <div class="chart">
                <h3>交易对净利润对比</h3>
                <img src="statistics_charts/symbol_net_profits.png" width="100%">
            </div>
            <div class="chart">
                <h3>日内交易时间分布</h3>
                <img src="statistics_charts/hourly_distribution.png" width="100%">
            </div>
            <div class="chart">
                <h3>每周交易日分布</h3>
                <img src="statistics_charts/daily_distribution.png" width="100%">
            </div>
            <div class="chart">
                <h3>交易盈亏分布</h3>
                <img src="statistics_charts/profit_distribution.png" width="100%">
            </div>
        </div>
    </body>
    </html>
    """

    # 写入HTML文件
    with open("trading_statistics_report.html", "w") as f:
        f.write(html)

    print(f"✅ 统计报告已生成: trading_statistics_report.html")
    return "trading_statistics_report.html"


def show_statistics(self):
    """显示交易统计信息"""
    # 加载持仓历史
    self._load_position_history()

    if not self.position_history:
        print("⚠️ 没有交易历史记录，无法生成统计")
        return

    print(f"📊 生成交易统计，共 {len(self.position_history)} 条记录")

    # 分析数据
    stats = self.analyze_position_statistics()

    # 生成图表
    self.generate_statistics_charts(stats)

    # 生成报告
    report_file = self.generate_statistics_report(stats)

    # 显示简要统计
    print("\n===== 交易统计摘要 =====")
    print(f"总交易: {stats['total_trades']} 次")
    print(f"盈利交易: {stats['winning_trades']} 次 ({stats['win_rate']:.2f}%)")
    print(f"亏损交易: {stats['losing_trades']} 次")
    print(f"总盈利: {stats['total_profit']:.2f}%")
    print(f"总亏损: {stats['total_loss']:.2f}%")
    print(f"净盈亏: {stats['total_profit'] - stats['total_loss']:.2f}%")
    print(f"盈亏比: {stats['profit_loss_ratio']:.2f}")
    print(f"平均持仓时间: {stats['avg_holding_time']:.2f} 小时")
    print(f"详细报告: {report_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='增强版交易机器人')
    parser.add_argument('--stats', action='store_true', help='生成交易统计报告')
    args = parser.parse_args()

    API_KEY = "lnfs30CvqF8cCIdRcIfW6kKnGGpLoRzTUrwdRslTX4e7a0O6OJ3SYsUT6gF1B26W"
    API_SECRET = "llSlxBLrrxh21ugMzli5x6NveNrwQyLBI7YEgTR4VOMyTmVP6V9uqmrN90hX10cn"

    bot = EnhancedTradingBot(API_KEY, API_SECRET, CONFIG)

    if args.stats:
        bot.show_statistics()
    else:
        bot.trade()