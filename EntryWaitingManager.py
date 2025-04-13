"""
入场时机等待模块：自动等待最佳入场时机
添加到EnhancedTradingBot类，实现异步价格监控与自动交易执行
"""
import threading
import time
import datetime
from logger_utils import Colors, print_colored


class EntryWaitingManager:
    """管理入场等待队列和价格监控"""

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
            entry_info: 包含入场信息的字典，至少应包含:
                - symbol: 交易对
                - side: 交易方向
                - amount: 交易金额
                - leverage: 杠杆倍数
                - target_price: 目标入场价格
                - expiry_time: 过期时间戳
                - entry_condition: 入场条件描述
        """
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

            # 确保监控线程在运行
            self.start_monitor()

            # 记录日志
            self.logger.info(f"添加入场等待: {entry_info['symbol']} {entry_info['side']}", extra={
                "target_price": entry_info['target_price'],
                "expiry_time": entry_info['expiry_time'],
                "entry_condition": entry_info.get('entry_condition', '未指定')
            })

    def remove_waiting_entry(self, symbol, side):
        """移除等待队列中的订单

        参数:
            symbol: 交易对
            side: 交易方向
        """
        with self.lock:
            before_len = len(self.waiting_entries)
            self.waiting_entries = [item for item in self.waiting_entries
                                    if not (item['symbol'] == symbol and item['side'] == side)]
            if len(self.waiting_entries) < before_len:
                print_colored(f"已从等待队列移除: {symbol} {side}", Colors.YELLOW)
                self.logger.info(f"移除入场等待: {symbol} {side}")

    def _price_monitor_loop(self):
        """价格监控循环，检查是否达到入场条件"""
        print_colored("🔄 入场价格监控循环已启动", Colors.BLUE)

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

                    # 检查是否达到入场条件
                    condition_met = False

                    if side == "BUY":
                        # 买入条件: 价格低于或等于目标价格
                        if current_price <= target_price * 1.001:  # 允许0.1%的误差
                            condition_met = True
                    else:  # SELL
                        # 卖出条件: 价格高于或等于目标价格
                        if current_price >= target_price * 0.999:  # 允许0.1%的误差
                            condition_met = True

                    # 如果达到条件，执行交易
                    if condition_met:
                        print_colored(
                            f"🎯 {symbol} {side} 达到入场条件! 目标价: {target_price:.6f}, 当前价: {current_price:.6f}",
                            Colors.GREEN + Colors.BOLD)
                        self.logger.info(f"{symbol} {side} 达到入场条件", extra={
                            "target_price": target_price,
                            "current_price": current_price
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
            time.sleep(5)  # 每5秒检查一次

        print_colored("🛑 入场价格监控循环已结束", Colors.YELLOW)


# 需要在EnhancedTradingBot类中的__init__方法中添加
# self.entry_manager = EntryWaitingManager(self)

# 需要修改place_futures_order_usdc方法，添加触发时机处理
def place_futures_order_usdc(self, symbol: str, side: str, amount: float, leverage: int = 5,
                             force_entry: bool = False) -> bool:
    """
    执行期货市场订单 - 增强版，支持入场时机等待

    参数:
        symbol: 交易对符号
        side: 交易方向 ('BUY' 或 'SELL')
        amount: 交易金额(USDC)
        leverage: 杠杆倍数
        force_entry: 是否强制入场，跳过入场时机检查

    返回:
        bool: 交易是否成功
    """
    import math
    import time
    from logger_utils import Colors, print_colored

    # 检查入场时机（除非是强制入场）
    if not force_entry:
        entry_timing = self.check_entry_timing(symbol, side)

        # 如果入场时机不佳，添加到等待队列
        if not entry_timing["should_enter"]:
            print_colored(f"\n⏳ {symbol} {side} 入场时机不佳，已加入等待队列", Colors.YELLOW)
            print_colored(f"预计更好的入场价格: {entry_timing['expected_price']:.6f}", Colors.YELLOW)
            print_colored(f"预计等待时间: {entry_timing['wait_minutes']} 分钟", Colors.YELLOW)
            print_colored(f"原因: {entry_timing['reason']}", Colors.YELLOW)

            # 计算过期时间（以分钟计）
            max_wait = min(entry_timing['wait_minutes'] * 1.5, 120)  # 最多等待预计时间的1.5倍或2小时
            expiry_time = time.time() + max_wait * 60

            # 添加到等待队列
            self.entry_manager.add_waiting_entry({
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'leverage': leverage,
                'target_price': entry_timing['expected_price'],
                'expiry_time': expiry_time,
                'entry_condition': entry_timing['reason'],
                'timing_quality': entry_timing['timing_quality']
            })

            return False  # 不立即执行交易，返回False表示暂未成功

    # 原有的交易逻辑...
    try:
        # 获取当前账户余额
        account_balance = self.get_futures_balance()
        print(f"📊 当前账户余额: {account_balance:.2f} USDC")

        # 省略原有代码...
        # 这里是原来的订单执行代码

        # 执行成功后，从等待队列中移除（如果存在）
        if hasattr(self, 'entry_manager'):
            self.entry_manager.remove_waiting_entry(symbol, side)

        return True
    except Exception as e:
        print_colored(f"❌ {symbol} {side} 交易过程中发生错误: {e}", Colors.ERROR)
        self.logger.error(f"{symbol} 交易错误", extra={"error": str(e)})
        return False


# 在交易机器人的初始化方法中添加等待管理器
def add_to_init(self):
    # 添加入场等待管理器
    self.entry_manager = EntryWaitingManager(self)
    print("✅ 入场时机等待管理器已初始化")


# 在trade方法中启动监控
def add_to_trade_method(self):
    # 启动入场时机监控
    if hasattr(self, 'entry_manager'):
        self.entry_manager.start_monitor()


# 添加查看当前等待队列的方法
def display_waiting_entries(self):
    """显示当前等待队列中的所有入场订单"""
    if not hasattr(self, 'entry_manager') or not self.entry_manager.waiting_entries:
        print("当前等待队列为空")
        return

    print("\n==== 入场等待队列 ====")
    print(f"{'交易对':<10} {'方向':<6} {'目标价':<12} {'当前价':<12} {'剩余时间':<10} {'条件':<40}")
    print("-" * 90)

    current_time = time.time()

    for entry in self.entry_manager.waiting_entries:
        symbol = entry['symbol']
        side = entry['side']
        target_price = entry['target_price']
        expiry_time = entry['expiry_time']
        condition = entry.get('entry_condition', '未指定')

        # 获取当前价格
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        except:
            current_price = 0.0

        # 计算剩余时间
        remaining_seconds = max(0, expiry_time - current_time)
        remaining_minutes = int(remaining_seconds / 60)
        remaining_seconds = int(remaining_seconds % 60)

        print(
            f"{symbol:<10} {side:<6} {target_price:<12.6f} {current_price:<12.6f} "
            f"{remaining_minutes:02d}:{remaining_seconds:02d} {condition[:40]}"
        )

    print("-" * 90)