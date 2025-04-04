"""
多时间框架协调系统整合示例
展示如何在交易机器人中整合多时间框架分析
"""
from indicators_module import calculate_optimized_indicators
from multi_timeframe_module import MultiTimeframeCoordinator
from logger_utils import Colors, print_colored
from quality_module import calculate_quality_score
import time


# 在交易机器人初始化中添加多时间框架协调器
def initialize_coordinator(self):
    """初始化多时间框架协调器"""
    self.mtf_coordinator = MultiTimeframeCoordinator(self.client, self.logger)
    self.logger.info("多时间框架协调器初始化完成")


# 修改原有的generate_trade_signal方法
def enhanced_generate_trade_signal(self, df, symbol):
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
            "timeframe_conflicts": coherence["trend_conflicts"]
        })

        # 返回信号和调整后的质量评分
        return signal, adjusted_score


# 在交易机器人的trade方法中整合多时间框架分析
def enhanced_trade_loop(self):
    """增强的交易循环，集成多时间框架分析"""
    print("启动增强版交易循环...")
    self.logger.info("增强版交易循环启动", extra={"version": "MTF-1.0"})

    # 确保多时间框架协调器已初始化
    if not hasattr(self, 'mtf_coordinator'):
        initialize_coordinator(self)

    while True:
        try:
            self.trade_cycle += 1
            print(f"\n==== 交易轮次 {self.trade_cycle} ====")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"当前时间: {current_time}")

            # 管理持仓
            if self.open_positions:
                self.manage_open_positions()

            # 获取账户余额
            account_balance = self.get_futures_balance()
            if account_balance < self.config.get("MIN_MARGIN_BALANCE", 10):
                print(f"⚠️ 账户余额不足: {account_balance} USDC")
                time.sleep(60)
                continue

            # 分析交易对并生成交易候选
            trade_candidates = []

            for symbol in self.config["TRADE_PAIRS"]:
                try:
                    print(f"\n分析交易对: {symbol}")
                    # 获取基础数据
                    df = self.get_historical_data_with_cache(symbol)
                    if df is None or df.empty:
                        continue

                    # 生成信号
                    signal, quality_score = self.enhanced_generate_trade_signal(df, symbol)

                    # 如果是有效信号，添加到候选列表
                    if signal in ["BUY", "SELL", "LIGHT_BUY", "LIGHT_SELL"]:
                        # 获取价格信息
                        current_data = self.client.futures_symbol_ticker(symbol=symbol)
                        current_price = float(current_data['price']) if current_data else None
                        if current_price is None:
                            continue

                        # 预测未来价格
                        predicted = self.predict_short_term_price(symbol, horizon_minutes=60)
                        if predicted is None:
                            continue

                        # 风险评估
                        risk = abs(current_price - predicted) / current_price

                        # 计算交易金额
                        if "LIGHT" in signal:
                            # 轻仓位交易
                            candidate_amount = self.calculate_dynamic_order_amount(risk, account_balance) * 0.5
                            print_colored(f"轻仓位交易: 标准金额的50%", Colors.YELLOW)
                        else:
                            # 标准仓位交易
                            candidate_amount = self.calculate_dynamic_order_amount(risk, account_balance)

                        # 添加到候选列表
                        trade_info = {
                            "symbol": symbol,
                            "signal": signal.replace("LIGHT_", ""),  # 移除LIGHT_前缀
                            "quality_score": quality_score,
                            "current_price": current_price,
                            "predicted_price": predicted,
                            "risk": risk,
                            "amount": candidate_amount,
                            "is_light": "LIGHT" in signal  # 标记是否轻仓位
                        }

                        trade_candidates.append(trade_info)

                        # 输出分析结果
                        signal_color = Colors.GREEN if "BUY" in signal else Colors.RED
                        print_colored(
                            f"{symbol} 信号: {signal_color}{signal}{Colors.RESET}, "
                            f"质量评分: {quality_score:.2f}, "
                            f"交易金额: {candidate_amount:.2f} USDC",
                            Colors.INFO
                        )

                except Exception as e:
                    self.logger.error(f"处理{symbol}时出错: {e}")
                    print(f"❌ 处理{symbol}时出错: {e}")

            # 按质量评分排序候选交易
            trade_candidates.sort(key=lambda x: x["quality_score"], reverse=True)

            # 执行交易
            executed_count = 0
            max_trades = self.config.get("MAX_PURCHASES_PER_ROUND", 3)

            for candidate in trade_candidates:
                if executed_count >= max_trades:
                    break

                symbol = candidate["symbol"]
                signal = candidate["signal"]
                amount = candidate["amount"]
                is_light = candidate["is_light"]

                print(f"\n🚀 执行交易: {symbol} {signal}")

                # 执行交易
                if self.place_futures_order_usdc(symbol, signal, amount):
                    executed_count += 1
                    print(f"✅ {symbol} {signal} 交易成功 "
                          f"({'轻仓位' if is_light else '标准仓位'})")
                else:
                    print(f"❌ {symbol} {signal} 交易失败")

            if executed_count == 0 and trade_candidates:
                print("⚠️ 本轮无成功执行的交易")
            elif not trade_candidates:
                print("📊 本轮无交易信号")

            # 显示持仓状态
            self.display_positions_status()

            # 等待下一轮
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


# 如何在交易机器人中整合
"""
在EnhancedTradingBot类中添加以下内容:

1. 在__init__方法中初始化多时间框架协调器:
   self.mtf_coordinator = MultiTimeframeCoordinator(self.client, self.logger)

2. 替换原有的generate_trade_signal方法为enhanced_generate_trade_signal

3. 替换原有的trade方法为enhanced_trade_loop或在原有trade方法中整合多时间框架分析
"""