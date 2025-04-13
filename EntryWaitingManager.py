"""
入场时机等待模块：自动等待最佳入场时机
添加到EnhancedTradingBot类，实现异步价格监控与自动交易执行
"""
import threading
import time
import datetime
from logger_utils import Colors, print_colored


class EntryWaitingManager:
    """管理入场等待队列和价格监控，支持高级SMC入场条件评估"""

    def __init__(self, trading_bot):
        """初始化入场等待管理器

        参数:
            trading_bot: 交易机器人实例，用于访问API和执行交易
        """
        self.trading_bot = trading_bot
        self.waiting_entries = []  # 等待执行的入场队列
        self.stop_flag = False  # 停止标志
        self.monitor_thread = None  # 价格监控线程
        self.lock = threading.Lock()  # 线程锁，防止竞争条件
        self.logger = trading_bot.logger  # 使用交易机器人的日志器
        self.check_interval = 5  # 默认每5秒检查一次
        self.deep_analysis_interval = 30  # 每30秒进行一次深度分析
        self.last_deep_analysis = {}  # 记录上次深度分析时间，格式: {symbol: timestamp}

    # [保留您原来的start_monitor和stop_monitor方法]

    def start_monitor(self):
        """启动价格监控线程"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_flag = False
            self.monitor_thread = threading.Thread(target=self._price_monitor_loop, daemon=True)
            self.monitor_thread.start()
            print_colored("✅ 入场时机监控线程已启动", Colors.GREEN)
            self.logger.info("入场时机监控线程已启动")

    def stop_monitor(self):
        """停止价格监控线程"""
        self.stop_flag = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
            print_colored("⏹️ 入场时机监控线程已停止", Colors.YELLOW)
            self.logger.info("入场时机监控线程已停止")

    def add_waiting_entry(self, entry_info):
        """添加等待执行的入场订单

        参数:
            entry_info: 包含入场信息的字典，除了原有字段外，还应包含:
                - initial_quality_score: 初始质量评分
                - min_quality_score: 最低执行质量分数
        """
        # 确保有初始质量评分
        if 'initial_quality_score' not in entry_info:
            entry_info['initial_quality_score'] = 5.0  # 默认中等质量

        # 确保有最低执行质量分数
        if 'min_quality_score' not in entry_info:
            entry_info['min_quality_score'] = 6.0  # 默认最低执行质量

        # 添加当前最佳质量评分字段，初始为初始质量评分
        entry_info['current_best_score'] = entry_info['initial_quality_score']

        # 添加上次分析时间
        entry_info['last_analysis_time'] = time.time()

        # 添加条件改善标志
        entry_info['conditions_improving'] = False

        # 调用原方法添加到等待队列
        with self.lock:
            # 检查是否已有相同交易对和方向的等待订单
            existing = next((item for item in self.waiting_entries
                             if item['symbol'] == entry_info['symbol'] and
                             item['side'] == entry_info['side']), None)

            if existing:
                # 如果已有相同订单，更新信息
                print_colored(f"更新 {entry_info['symbol']} {entry_info['side']} 的等待入场信息", Colors.YELLOW)
                self.waiting_entries.remove(existing)

            # 添加到等待队列
            self.waiting_entries.append(entry_info)
            print_colored(f"添加到入场等待队列: {entry_info['symbol']} {entry_info['side']}", Colors.CYAN)
            print_colored(
                f"目标价格: {entry_info['target_price']:.6f}, 过期时间: {datetime.datetime.fromtimestamp(entry_info['expiry_time']).strftime('%H:%M:%S')}",
                Colors.CYAN)
            print_colored(
                f"初始质量评分: {entry_info['initial_quality_score']:.2f}, 最低执行分数: {entry_info['min_quality_score']:.2f}",
                Colors.CYAN)

            # 确保监控线程在运行
            self.start_monitor()

            # 记录日志
            self.logger.info(f"添加入场等待: {entry_info['symbol']} {entry_info['side']}", extra={
                "target_price": entry_info['target_price'],
                "expiry_time": entry_info['expiry_time'],
                "entry_condition": entry_info.get('entry_condition', '未指定'),
                "initial_quality_score": entry_info['initial_quality_score']
            })

    def _price_monitor_loop(self):
        """价格监控循环，检查是否达到入场条件，包括高级SMC入场分析"""
        print_colored("🔄 入场价格监控循环已启动", Colors.BLUE)
        last_check_time = time.time()

        while not self.stop_flag:
            current_time = time.time()
            executed_entries = []
            expired_entries = []

            # 使用锁复制列表，避免迭代时修改
            with self.lock:
                entries_to_check = self.waiting_entries.copy()

            for entry in entries_to_check:
                symbol = entry['symbol']
                side = entry['side']
                target_price = entry['target_price']
                expiry_time = entry['expiry_time']

                # 检查是否过期
                if current_time > expiry_time:
                    print_colored(f"⏱️ {symbol} {side} 入场等待已过期", Colors.YELLOW)
                    expired_entries.append((symbol, side))
                    continue

                try:
                    # 获取当前价格
                    ticker = self.trading_bot.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])

                    # 基础价格条件检查
                    price_condition_met = self._check_price_condition(side, current_price, target_price)

                    # 定期进行深度分析，或者当价格接近目标价格时进行
                    should_deep_analyze = (
                            current_time - entry.get('last_analysis_time', 0) >= self.deep_analysis_interval or
                            abs(current_price - target_price) / target_price < 0.005  # 距离目标价格0.5%以内
                    )

                    entry_conditions_met = False
                    entry_reason = ""

                    # 如果需要进行深度分析，评估完整的入场条件
                    if should_deep_analyze:
                        entry['last_analysis_time'] = current_time
                        entry_analysis = self._analyze_entry_conditions(symbol, side, current_price, entry)

                        # 更新入场条件状态
                        entry_conditions_met = entry_analysis['should_enter']
                        entry_reason = entry_analysis['reason']

                        # 更新质量评分
                        if entry_analysis['quality_score'] > entry['current_best_score']:
                            entry['current_best_score'] = entry_analysis['quality_score']
                            entry['conditions_improving'] = True
                            print_colored(
                                f"📈 {symbol} {side} 入场条件改善，质量评分: {entry_analysis['quality_score']:.2f}",
                                Colors.GREEN
                            )
                        else:
                            entry['conditions_improving'] = False

                    # 确定是否应该入场
                    should_enter = False

                    # 情况1: 分析结果直接建议入场
                    if entry_conditions_met:
                        should_enter = True
                        reason = entry_reason

                    # 情况2: 价格条件满足且已有足够高的质量评分
                    elif price_condition_met and entry['current_best_score'] >= entry['min_quality_score']:
                        should_enter = True
                        reason = f"价格条件满足，质量评分充分 ({entry['current_best_score']:.2f})"

                    # 情况3: 入场条件持续改善且已有较高的质量评分
                    elif entry['conditions_improving'] and entry['current_best_score'] >= 7.0:
                        should_enter = True
                        reason = f"入场条件持续改善，质量评分高 ({entry['current_best_score']:.2f})"

                    # 如果应该入场，执行交易
                    if should_enter:
                        print_colored(
                            f"🎯 {symbol} {side} 达到入场条件! 目标价: {target_price:.6f}, 当前价: {current_price:.6f}",
                            Colors.GREEN + Colors.BOLD
                        )
                        print_colored(f"入场原因: {reason}", Colors.GREEN)
                        self.logger.info(f"{symbol} {side} 达到入场条件", extra={
                            "target_price": target_price,
                            "current_price": current_price,
                            "reason": reason,
                            "quality_score": entry['current_best_score']
                        })

                        # 执行交易
                        success = self.trading_bot.place_futures_order_usdc(
                            symbol=symbol,
                            side=side,
                            amount=entry['amount'],
                            leverage=entry['leverage'],
                            force_entry=True  # 使用强制入场标志，绕过入场检查
                        )

                        if success:
                            print_colored(f"✅ {symbol} {side} 条件触发交易执行成功!", Colors.GREEN + Colors.BOLD)
                            executed_entries.append((symbol, side))
                        else:
                            print_colored(f"❌ {symbol} {side} 条件触发但交易执行失败", Colors.RED)
                            # 失败也从队列中移除，避免反复尝试失败的交易
                            executed_entries.append((symbol, side))

                except Exception as e:
                    print_colored(f"监控 {symbol} 价格时出错: {e}", Colors.ERROR)
                    self.logger.error(f"价格监控错误: {symbol}", extra={"error": str(e)})

            # 移除已执行或过期的条目
            with self.lock:
                for symbol, side in executed_entries + expired_entries:
                    self.remove_waiting_entry(symbol, side)

                # 如果队列为空，可以考虑停止监控线程以节省资源
                if not self.waiting_entries:
                    print_colored("入场等待队列为空，监控将在下一轮后暂停", Colors.YELLOW)
                    self.stop_flag = True

            # 睡眠一段时间再检查
            time.sleep(self.check_interval)

        print_colored("🛑 入场价格监控循环已结束", Colors.YELLOW)

    def _check_price_condition(self, side, current_price, target_price):
        """检查价格是否满足入场条件

        参数:
            side: 交易方向 ('BUY' 或 'SELL')
            current_price: 当前价格
            target_price: 目标价格

        返回:
            bool: 价格条件是否满足
        """
        if side == "BUY":
            # 买入条件: 价格低于或等于目标价格
            return current_price <= target_price * 1.001  # 允许0.1%的误差
        else:  # SELL
            # 卖出条件: 价格高于或等于目标价格
            return current_price >= target_price * 0.999  # 允许0.1%的误差

    def _analyze_entry_conditions(self, symbol, side, current_price, entry_info):
        """分析完整入场条件，包括价格位置、订单块、斐波那契水平等

        参数:
            symbol: 交易对符号
            side: 交易方向
            current_price: 当前价格
            entry_info: 入场信息字典

        返回:
            dict: 包含入场分析结果
        """
        try:
            # 获取最新数据
            df = self.trading_bot.get_historical_data_with_cache(symbol, force_refresh=True)

            if df is None or len(df) < 20:
                return {
                    "should_enter": False,
                    "reason": "数据不足，无法分析",
                    "quality_score": 0.0
                }

            # 确保数据中包含必要的指标
            if 'ATR' not in df.columns:
                # 尝试计算ATR
                if 'TR' not in df.columns and len(df) >= 14:
                    high = df['high']
                    low = df['low']
                    close = df['close'].shift(1)

                    # 计算TR
                    tr1 = high - low
                    tr2 = abs(high - close)
                    tr3 = abs(low - close)
                    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

                    # 计算ATR
                    df['ATR'] = tr.rolling(window=14).mean()

            # 分析价格位置条件
            # 这里可以应用前面定义的价格位置分析函数
            # 如果您已经有这些函数，可以调用它们
            # 如果没有，可以添加简化版本

            # 简化版价格位置分析
            position_score = self._analyze_price_position(df, current_price, side)

            # 确定是否应该入场
            should_enter = False
            reason = ""

            if position_score >= 7.0:
                should_enter = True
                reason = f"价格位置条件优秀 (评分: {position_score:.2f}/10)"
            elif position_score >= 6.0 and self._check_price_condition(side, current_price, entry_info['target_price']):
                should_enter = True
                reason = f"价格位置良好且已达到目标价格 (评分: {position_score:.2f}/10)"
            elif position_score > entry_info['initial_quality_score'] + 1.0:
                should_enter = True
                reason = f"入场条件显著改善 (从 {entry_info['initial_quality_score']:.2f} 提高到 {position_score:.2f})"
            else:
                reason = f"入场条件尚未满足 (当前评分: {position_score:.2f}/10)"

            return {
                "should_enter": should_enter,
                "reason": reason,
                "quality_score": position_score
            }

        except Exception as e:
            print_colored(f"分析 {symbol} 入场条件时出错: {e}", Colors.ERROR)
            self.logger.error(f"入场条件分析错误: {symbol}", extra={"error": str(e)})
            return {
                "should_enter": False,
                "reason": f"分析过程出错: {str(e)}",
                "quality_score": entry_info['initial_quality_score']
            }

    def _analyze_price_position(self, df, current_price, side):
        """分析价格位置质量

        参数:
            df: 价格数据DataFrame
            current_price: 当前价格
            side: 交易方向

        返回:
            float: 价格位置质量评分 (0-10)
        """
        # 基础评分
        score = 5.0

        # 分析均线位置
        if 'EMA20' in df.columns and 'EMA50' in df.columns:
            ema20 = df['EMA20'].iloc[-1]
            ema50 = df['EMA50'].iloc[-1]

            if side == "BUY":
                # 买入时，价格在均线上方为好
                if current_price > ema20 > ema50:
                    score += 1.0  # 多头排列
                elif current_price > ema20:
                    score += 0.5  # 价格在短期均线上方
                elif abs(current_price - ema20) / current_price < 0.005:
                    score += 0.8  # 价格接近短期均线，可能是支撑
            else:  # SELL
                # 卖出时，价格在均线下方为好
                if current_price < ema20 < ema50:
                    score += 1.0  # 空头排列
                elif current_price < ema20:
                    score += 0.5  # 价格在短期均线下方
                elif abs(current_price - ema20) / current_price < 0.005:
                    score += 0.8  # 价格接近短期均线，可能是阻力

        # 分析布林带位置
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_middle = df['BB_Middle'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]

            if side == "BUY":
                # 买入时，接近下轨为好
                if current_price < bb_lower * 1.01:
                    score += 1.5  # 接近或低于下轨
                elif current_price < bb_middle * 0.98:
                    score += 0.8  # 中轨和下轨之间，偏下
            else:  # SELL
                # 卖出时，接近上轨为好
                if current_price > bb_upper * 0.99:
                    score += 1.5  # 接近或高于上轨
                elif current_price > bb_middle * 1.02:
                    score += 0.8  # 中轨和上轨之间，偏上

        # 分析RSI位置
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]

            if side == "BUY" and rsi < 30:
                score += 1.5  # RSI超卖
            elif side == "SELL" and rsi > 70:
                score += 1.5  # RSI超买

        # 检查ATR，确保不在剧烈波动中
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            atr_mean = df['ATR'].rolling(window=20).mean().iloc[-1]

            if atr < atr_mean * 0.8:
                score += 0.7  # 波动性低于平均，可能更稳定
            elif atr > atr_mean * 1.5:
                score -= 1.0  # 波动性过高，风险增加

        # 确保分数在0-10范围内
        return max(0, min(10, score))