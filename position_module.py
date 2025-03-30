import pandas as pd
import numpy as np
import time

def load_positions(client, logger=None):
    """
    加载当前所有持仓

    参数:
        client: Binance客户端
        logger: 日志对象（可选）

    返回:
        open_positions: 持仓列表
    """
    try:
        positions = client.futures_position_information()
        open_positions = []
        for pos in positions:
            amt = float(pos.get("positionAmt", 0))
            if abs(amt) > 0:
                position_side = pos.get("positionSide", "BOTH")
                # 处理仓位方向
                if position_side == "BOTH":
                    side = "BUY" if amt > 0 else "SELL"
                else:
                    side = position_side  # LONG 或 SHORT

                open_positions.append({
                    "symbol": pos["symbol"],
                    "side": side,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "quantity": abs(amt),
                    "open_time": float(pos.get("updateTime", 0)) / 1000,
                    "max_profit": 0.0,
                    "position_side": position_side,
                    "target_profit": 0.0,
                    "dynamic_take_profit": 0.06,  # 默认6%止盈
                    "stop_loss": -0.03,  # 默认3%止损
                    "last_check_time": time.time(),
                    "last_check_price": float(pos.get("markPrice", 0)),
                    "position_id": f"{pos['symbol']}_{position_side}_{int(time.time())}"
                })

                if logger:
                    logger.info(f"加载持仓: {pos['symbol']} {side} {amt}")
        return open_positions
    except Exception as e:
        if logger:
            logger.error(f"加载持仓失败: {e}")
        return []

def get_total_position_exposure(positions, account_balance):
    """
    计算当前总持仓占账户余额的百分比

    参数:
        positions: 持仓列表
        account_balance: 账户总余额

    返回:
        total_exposure: 总持仓占比（百分比）
        symbol_exposures: 每个交易对的持仓占比字典
    """
    if account_balance <= 0:
        return 100.0, {}  # 防止除零错误

    # 计算总持仓价值
    total_position_value = 0
    symbol_values = {}

    for pos in positions:
        position_value = pos["entry_price"] * pos["quantity"]
        total_position_value += position_value

        # 累加每个交易对的持仓价值
        symbol = pos["symbol"]
        if symbol in symbol_values:
            symbol_values[symbol] += position_value
        else:
            symbol_values[symbol] = position_value

    # 计算总持仓占比
    total_exposure = (total_position_value / account_balance) * 100

    # 计算每个交易对的持仓占比
    symbol_exposures = {s: (v / account_balance) * 100 for s, v in symbol_values.items()}

    return total_exposure, symbol_exposures


def calculate_order_amount(account_balance, symbol_exposure, symbol=None,
                           max_total_exposure=85, max_symbol_exposure=15,
                           default_order_pct=5):
    """
    计算适当的下单金额，考虑高价值货币的特殊处理

    参数:
        account_balance: 账户总余额
        symbol_exposure: 当前交易对的持仓占比（百分比）
        symbol: 交易对名称，用于识别高价值货币
        max_total_exposure: 最大总持仓比例（默认85%）
        max_symbol_exposure: 单一货币最大持仓比例（默认15%）
        default_order_pct: 默认下单比例（账户的5%）

    返回:
        order_amount: 建议下单金额
        order_pct: 实际下单比例
    """
    # 确保账户有余额
    if account_balance <= 0:
        print(f"⚠️ 账户余额为零，无法下单")
        return 0, 0

    # 高价值货币使用更小的下单比例
    high_value_symbols = ['BTCUSDT', 'ETHUSDT', 'PAXGUSDT', 'WBTCUSDT', 'WETHUSDT']
    if symbol in high_value_symbols:
        adjusted_order_pct = 1.0  # 对高价值货币使用1%而不是5%
        print(f"📉 {symbol}是高价值货币，下单比例调整为{adjusted_order_pct}%（默认{default_order_pct}%）")
    else:
        adjusted_order_pct = default_order_pct

    # 默认下单金额
    target_amount = account_balance * (adjusted_order_pct / 100)
    print(f"📊 目标下单金额: {target_amount:.2f} USDC ({adjusted_order_pct}% 账户余额)")

    # 检查是否会超过单一货币限制
    remaining_symbol_exposure = max_symbol_exposure - symbol_exposure
    max_symbol_amount = account_balance * (remaining_symbol_exposure / 100)

    # 如果剩余额度不足，调整下单金额
    if target_amount > max_symbol_amount:
        if max_symbol_amount <= 0:
            # 已达到该币种限制
            print(f"⚠️ {symbol}敞口已达到上限 {max_symbol_exposure}%，无法下单")
            return 0, 0

        # 调整为最大可用金额
        order_amount = max_symbol_amount
        order_pct = remaining_symbol_exposure
        print(f"⚠️ 调整下单比例：从 {adjusted_order_pct}% 减少至 {order_pct:.2f}%")
    else:
        order_amount = target_amount
        order_pct = adjusted_order_pct

    # 确保有足够的金额但不会太小
    min_order_amount = 5  # 降低最小下单额至5美元
    if order_amount < min_order_amount:
        if max_symbol_amount >= min_order_amount:
            order_amount = min_order_amount
            order_pct = (order_amount / account_balance) * 100
            print(f"⚠️ 下单金额小于最小值，调整至: {min_order_amount} USDC")
        else:
            # 金额太小，不下单
            print(f"⚠️ 计算的下单金额 ({order_amount:.2f} USDC) 小于最小值，无法下单")
            return 0, 0

    # 打印最终下单信息
    print(f"📈 最终下单金额: {order_amount:.2f} USDC ({order_pct:.2f}% 账户余额)")

    return order_amount, order_pct

def adjust_position_for_market_change(positions, client, logger=None):
    """
    根据市场变化调整持仓状态

    参数:
        positions: 持仓列表
        client: Binance客户端
        logger: 日志对象（可选）

    返回:
        updated_positions: 更新后的持仓列表
        actions: 执行的动作列表
    """
    if not positions:
        return [], []

    actions = []
    updated_positions = positions.copy()
    current_time = time.time()

    for i, pos in enumerate(updated_positions):
        symbol = pos["symbol"]
        entry_price = pos["entry_price"]
        side = pos["side"]
        quantity = pos["quantity"]
        max_profit = pos.get("max_profit", 0)
        open_time = pos.get("open_time", current_time)
        holding_minutes = (current_time - open_time) / 60

        # 获取当前价格
        try:
            ticker = client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        except Exception as e:
            if logger:
                logger.error(f"获取{symbol}价格失败: {e}")
            continue

        # 计算当前利润
        if side == "LONG" or side == "BUY":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # SHORT 或 SELL
            profit_pct = (entry_price - current_price) / entry_price

        profit_amount = profit_pct * quantity * entry_price

        # 更新最大利润
        if profit_amount > max_profit:
            updated_positions[i]["max_profit"] = profit_amount
            max_profit = profit_amount

        # 动态止盈止损计算
        dynamic_take_profit = pos.get("dynamic_take_profit", 0.06)  # 默认6%
        stop_loss = pos.get("stop_loss", -0.03)  # 默认-3%

        # 根据持仓时间和最大利润调整止盈止损
        if holding_minutes > 60:  # 持仓超过1小时
            # 提高止盈点以锁定利润
            if max_profit > 0 and profit_pct > 0:
                # 根据最大利润调整止盈
                max_profit_pct = max_profit / (quantity * entry_price)
                if max_profit_pct > 0.10:  # 最大利润超过10%
                    new_take_profit = max(dynamic_take_profit, max_profit_pct * 0.7)  # 锁定70%的最大利润
                    updated_positions[i]["dynamic_take_profit"] = new_take_profit
                    if logger:
                        logger.info(f"{symbol} {side} 调整止盈至 {new_take_profit:.2%}")

            # 调整止损
            if profit_pct > 0.05:  # 利润超过5%
                # 将止损移至保本线
                updated_positions[i]["stop_loss"] = 0.001  # 略高于保本
                if logger:
                    logger.info(f"{symbol} {side} 调整止损至保本线")

        # 检查是否需要止盈或止损
        action = None
        if profit_pct >= dynamic_take_profit:
            action = "take_profit"
        elif profit_pct <= stop_loss:
            action = "stop_loss"
        elif holding_minutes > 240 and profit_pct < 0:  # 4小时后仍亏损
            action = "time_stop"

        if action:
            actions.append({
                "symbol": symbol,
                "side": side,
                "action": action,
                "profit_pct": profit_pct,
                "profit_amount": profit_amount,
                "holding_minutes": holding_minutes
            })

        # 更新持仓检查时间和价格
        updated_positions[i]["last_check_time"] = current_time
        updated_positions[i]["last_check_price"] = current_price

    return updated_positions, actions