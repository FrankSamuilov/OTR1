import math
import time
import numpy as np
import pandas as pd
from binance.exceptions import BinanceAPIException


def get_max_leverage(client, symbol, max_allowed=20):
    """
    获取交易对的最大允许杠杆，确保不超过20倍的限制

    参数:
        client: Binance客户端
        symbol: 交易对
        max_allowed: 最大允许杠杆，默认20

    返回:
        最大可用杠杆
    """
    try:
        # 获取杠杆分层信息
        leverage_brackets = client.futures_leverage_bracket(symbol=symbol)
        if not leverage_brackets:
            print(f"⚠️ {symbol} 无法获取杠杆分层信息，使用默认杠杆5倍")
            return min(5, max_allowed)

        # 找到对应的交易对信息
        bracket_info = None
        for item in leverage_brackets:
            if item['symbol'] == symbol:
                bracket_info = item['brackets']
                break

        if not bracket_info:
            print(f"⚠️ {symbol} 无法找到杠杆分层信息，使用默认杠杆5倍")
            return min(5, max_allowed)

        # 获取最大杠杆
        max_leverage = bracket_info[0]['initialLeverage']  # 第一层通常是最大杠杆
        capped_leverage = min(max_leverage, max_allowed)
        print(f"🔍 {symbol} 最大杠杆: {max_leverage}倍，限制后: {capped_leverage}倍")
        return capped_leverage
    except Exception as e:
        print(f"❌ 获取{symbol}杠杆信息失败: {e}")
        return min(5, max_allowed)  # 出错时返回默认值


def calculate_dynamic_leverage(client, symbol, quality_score, trend, market_conditions, max_allowed=20):
    """
    基于市场情况和质量评分动态计算合适的杠杆倍数

    参数:
        client: Binance客户端
        symbol: 交易对
        quality_score: 质量评分 (0-10)
        trend: 市场趋势 ("UP", "DOWN", "NEUTRAL")
        market_conditions: 市场环境信息
        max_allowed: 最大允许杠杆

    返回:
        leverage: 计算的杠杆倍数
    """
    # 获取交易对最大可用杠杆
    max_leverage = get_max_leverage(client, symbol, max_allowed)

    # 基础杠杆 - 基于质量评分
    if quality_score >= 9.0:  # 极高质量
        base_leverage = 20
        print(f"📈 {symbol} 极高质量评分 ({quality_score:.2f})，基础杠杆: 20倍")
    elif quality_score >= 8.0:
        base_leverage = 15
        print(f"📈 {symbol} 非常高质量评分 ({quality_score:.2f})，基础杠杆: 15倍")
    elif quality_score >= 7.0:
        base_leverage = 10
        print(f"📈 {symbol} 高质量评分 ({quality_score:.2f})，基础杠杆: 10倍")
    elif quality_score >= 6.0:
        base_leverage = 7
        print(f"📈 {symbol} 良好质量评分 ({quality_score:.2f})，基础杠杆: 7倍")
    elif quality_score >= 5.0:
        base_leverage = 5
        print(f"📈 {symbol} 中等质量评分 ({quality_score:.2f})，基础杠杆: 5倍")
    else:
        base_leverage = 3
        print(f"📈 {symbol} 较低质量评分 ({quality_score:.2f})，基础杠杆: 3倍")

    # 市场趋势调整
    trend_multiplier = 1.0
    if trend == "UP" or trend == "DOWN":
        trend_multiplier = 1.2  # 明确趋势加大杠杆
        print(f"📈 {symbol} 明确{trend}趋势，杠杆乘数: +20%")
    elif trend == "NEUTRAL":
        trend_multiplier = 0.7  # 中性趋势降低杠杆
        print(f"📈 {symbol} 中性趋势，杠杆乘数: -30%")

    # 市场环境调整
    env_multiplier = 1.0
    if market_conditions and 'environment' in market_conditions:
        env = market_conditions['environment']
        if env == 'trending':
            env_multiplier = 1.1  # 趋势市场略微增加杠杆
            print(f"📈 {symbol} 趋势市场环境，杠杆乘数: +10%")
        elif env == 'ranging':
            env_multiplier = 0.6  # 震荡市场大幅减少杠杆
            print(f"📈 {symbol} 震荡市场环境，杠杆乘数: -40%")
        elif env == 'breakout':
            env_multiplier = 1.2  # 突破市场增加杠杆
            print(f"📈 {symbol} 突破市场环境，杠杆乘数: +20%")
        elif env == 'extreme_volatility':
            env_multiplier = 0.4  # 极端波动市场大幅减少杠杆
            print(f"📈 {symbol} 极端波动市场，杠杆乘数: -60%")

    # 计算最终杠杆，并确保在允许范围内
    final_leverage = max(1, min(max_leverage, round(base_leverage * trend_multiplier * env_multiplier)))
    print(f"🎯 {symbol} 最终杠杆: {final_leverage}倍")
    return final_leverage


def get_precise_quantity(client, symbol, quantity):
    """
    根据交易所规则，获取精确的交易数量

    参数:
        client: Binance客户端
        symbol: 交易对
        quantity: 原始数量

    返回:
        调整后的精确数量
    """
    try:
        # 获取交易所信息
        info = client.futures_exchange_info()

        # 查找该交易对的数量精度
        for item in info['symbols']:
            if item['symbol'] == symbol:
                for f in item['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        min_qty = float(f['minQty'])
                        max_qty = float(f['maxQty'])
                        step_size = float(f['stepSize'])

                        # 调整数量到步长的整数倍
                        quantity = max(min_qty, min(max_qty, quantity))
                        precision = int(round(-math.log(step_size, 10), 0))
                        quantity = round(math.floor(quantity * 10 ** precision) / 10 ** precision, precision)

                        print(f"🔢 {symbol} 调整数量: {quantity} (最小:{min_qty}, 最大:{max_qty}, 步长:{step_size})")
                        return quantity

        # 如果没有找到精度信息，返回原始数量
        print(f"⚠️ {symbol} 无法获取数量精度信息")
        return round(quantity, 4)
    except Exception as e:
        print(f"❌ 获取精确数量失败: {e}")
        return round(quantity, 4)  # 出错时使用默认精度


def format_quantity(self, symbol, quantity):
    """
    格式化交易数量，确保符合交易所要求

    参数:
        symbol: 交易对符号
        quantity: 原始数量

    返回:
        格式化后的数量字符串
    """
    try:
        # 获取交易对信息
        info = self.client.futures_exchange_info()

        # 默认精度（如果无法获取特定交易对信息）
        precision = 3

        # 查找该交易对的精度信息
        for item in info['symbols']:
            if item['symbol'] == symbol:
                for f in item['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        precision = int(round(-math.log(step_size, 10), 0)) if step_size < 1 else 0
                break

        # 四舍五入到适当精度
        formatted_quantity = round(float(quantity), precision)

        # 转为字符串，避免科学计数法
        if precision > 0:
            quantity_str = f"{{:.{precision}f}}".format(formatted_quantity)
        else:
            quantity_str = str(int(formatted_quantity))

        return quantity_str

    except Exception as e:
        print(f"❌ 格式化数量出错 ({symbol}, {quantity}): {e}")
        # 作为后备方案，尝试简单格式化
        try:
            # 尝试使用最基本的格式化，去除小数点后的零
            return str(float(quantity)).rstrip('0').rstrip('.') if '.' in str(float(quantity)) else str(int(quantity))
        except:
            # 如果还是失败，直接返回原始数量的字符串
            return str(quantity)


def adjust_quantity_for_leverage(quantity, leverage, current_price, account_balance, max_risk_pct=20.0):
    """
    根据杠杆调整交易数量，确保不会超过账户最大风险承受能力

    参数:
        quantity: 原始数量
        leverage: 杠杆倍数
        current_price: 当前价格
        account_balance: 账户余额
        max_risk_pct: 最大风险百分比（账户的百分比）

    返回:
        调整后的数量
    """
    # 计算当前交易价值
    trade_value = quantity * current_price

    # 计算实际风险金额（考虑杠杆）
    risk_amount = trade_value / leverage

    # 计算最大允许风险金额
    max_risk_amount = account_balance * (max_risk_pct / 100)

    # 如果风险金额超过最大允许风险，调整数量
    if risk_amount > max_risk_amount:
        adjusted_quantity = (max_risk_amount * leverage) / current_price
        print(f"⚠️ 风险控制: 数量从 {quantity} 减少至 {adjusted_quantity} (最大风险: {max_risk_pct}%)")
        return adjusted_quantity

    return quantity


def get_order_book_depth(client, symbol, limit=10):
    """
    获取交易对的订单簿数据，用于分析市场深度

    参数:
        client: Binance客户端
        symbol: 交易对
        limit: 深度级别

    返回:
        order_book: 订单簿信息
    """
    try:
        order_book = client.futures_order_book(symbol=symbol, limit=limit)

        # 计算买卖压力比
        total_bid_qty = sum(float(item[1]) for item in order_book['bids'])
        total_ask_qty = sum(float(item[1]) for item in order_book['asks'])

        bid_ask_ratio = total_bid_qty / total_ask_qty if total_ask_qty > 0 else float('inf')

        # 分析价格距离
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        spread = (best_ask - best_bid) / best_bid * 100  # 价差百分比

        # 返回深度分析结果
        depth_analysis = {
            'bid_ask_ratio': bid_ask_ratio,
            'spread_pct': spread,
            'top_bids': order_book['bids'][:5],
            'top_asks': order_book['asks'][:5],
            'buy_pressure': total_bid_qty,
            'sell_pressure': total_ask_qty
        }

        print(f"📊 {symbol} 订单簿分析 - 买卖比: {bid_ask_ratio:.2f}, 价差: {spread:.3f}%")
        return depth_analysis
    except Exception as e:
        print(f"❌ 获取订单簿数据失败: {e}")
        return None


def place_dual_orders(client, symbol, primary_side, quality_score, account_balance, logger=None,
                      leverage=5, secondary_size_pct=0.3, max_risk_pct=20.0):
    """
    根据交易信号同时下多空双向订单，用于震荡市场或不确定趋势

    参数:
        client: Binance客户端
        symbol: 交易对
        primary_side: 主要方向 ('BUY' or 'SELL')
        quality_score: 质量评分 (0-10)
        account_balance: 账户余额
        logger: 日志对象（可选）
        leverage: 杠杆倍数
        secondary_size_pct: 次要方向订单大小占主要方向的百分比
        max_risk_pct: 最大风险百分比

    返回:
        success: 是否成功下单
        orders: 订单信息
    """
    try:
        print(f"🔄 {symbol} 尝试下双向订单 - 主方向: {primary_side}, 杠杆: {leverage}倍")

        # 获取当前价格
        ticker = client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])

        # 计算主订单数量
        position_value = account_balance * (max_risk_pct / 100) * 0.8  # 使用80%的风险额度给主订单
        main_quantity = position_value / current_price

        # 精确化数量
        main_quantity = get_precise_quantity(client, symbol, main_quantity)

        # 计算次订单数量
        secondary_quantity = main_quantity * secondary_size_pct
        secondary_quantity = get_precise_quantity(client, symbol, secondary_quantity)

        # 设置杠杆
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"✅ {symbol} 设置杠杆成功: {leverage}倍")
        except BinanceAPIException as e:
            if "leverage not modified" not in str(e).lower():
                print(f"⚠️ {symbol} 设置杠杆失败: {e}")
                if logger:
                    logger.warning(f"{symbol}设置杠杆失败: {e}")

        # 计算反向订单方向
        secondary_side = "SELL" if primary_side == "BUY" else "BUY"

        # 执行主订单
        main_order = client.futures_create_order(
            symbol=symbol,
            side=primary_side,
            type="MARKET",
            quantity=str(main_quantity),
            positionSide="LONG" if primary_side == "BUY" else "SHORT"
        )

        if logger:
            logger.info(f"{symbol} {primary_side} 主订单执行成功", extra={
                "order_id": main_order.get("orderId", "unknown"),
                "quantity": main_quantity,
                "leverage": leverage
            })

        print(f"✅ {symbol} {primary_side} 主订单执行成功, 数量: {main_quantity}")

        # 等待一秒避免API速率限制
        time.sleep(1)

        # 执行次订单
        secondary_order = client.futures_create_order(
            symbol=symbol,
            side=secondary_side,
            type="MARKET",
            quantity=str(secondary_quantity),
            positionSide="LONG" if secondary_side == "BUY" else "SHORT"
        )

        if logger:
            logger.info(f"{symbol} {secondary_side} 次订单执行成功", extra={
                "order_id": secondary_order.get("orderId", "unknown"),
                "quantity": secondary_quantity,
                "leverage": leverage
            })

        print(f"✅ {symbol} {secondary_side} 次订单执行成功, 数量: {secondary_quantity}")

        # 返回订单信息
        return True, {
            "main_order": main_order,
            "secondary_order": secondary_order,
            "main_quantity": main_quantity,
            "secondary_quantity": secondary_quantity,
            "primary_side": primary_side,
            "leverage": leverage
        }

    except Exception as e:
        print(f"❌ {symbol} 双向订单执行失败: {e}")
        if logger:
            logger.error(f"{symbol} 双向订单执行失败: {e}")
        return False, None


def place_smart_order(client, symbol, side, quantity, leverage=5, current_price=None, logger=None):
    """
    智能下单函数，自动选择最佳订单类型和参数

    参数:
        client: Binance客户端
        symbol: 交易对
        side: 交易方向 ('BUY' or 'SELL')
        quantity: 交易数量
        leverage: 杠杆倍数
        current_price: 当前价格（如果已知）
        logger: 日志对象（可选）

    返回:
        success: 是否成功下单
        order_info: 订单信息
    """
    try:
        # 获取当前价格（如果未提供）
        if current_price is None:
            ticker = client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

        # 分析订单簿深度
        depth = get_order_book_depth(client, symbol)

        # 设置杠杆
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"✅ {symbol} 设置杠杆成功: {leverage}倍")
        except BinanceAPIException as e:
            if "leverage not modified" not in str(e).lower():
                print(f"⚠️ {symbol} 设置杠杆失败: {e}")
                if logger:
                    logger.warning(f"{symbol}设置杠杆失败: {e}")

        # 确定订单类型和参数
        order_type = "MARKET"  # 默认使用市价单
        order_params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
            "positionSide": "LONG" if side == "BUY" else "SHORT"
        }

        # 如果市场深度良好，且价差较小，可以考虑限价单
        if depth and depth['spread_pct'] < 0.05 and depth['bid_ask_ratio'] > 0.8 and depth['bid_ask_ratio'] < 1.2:
            # 市场比较平衡，可以尝试限价单以获得更好的成交价格
            limit_price = current_price * (0.9995 if side == "BUY" else 1.0005)  # 略好于市价
            print(f"📊 {symbol} 市场深度良好，使用限价单, 价格: {limit_price:.6f}")

            # 更新为限价单参数
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "quantity": str(quantity),
                "timeInForce": "GTC",
                "price": str(round(limit_price, 6)),
                "positionSide": "LONG" if side == "BUY" else "SHORT"
            }
        else:
            print(f"📊 {symbol} 使用市价单")

        # 执行订单
        order = client.futures_create_order(**order_params)

        print(f"✅ {symbol} {side} 订单执行成功, 数量: {quantity}, 类型: {order_params['type']}")
        if logger:
            logger.info(f"{symbol} {side} 订单执行成功", extra={
                "order_id": order.get("orderId", "unknown"),
                "quantity": quantity,
                "leverage": leverage,
                "order_type": order_params['type']
            })

        return True, {
            "order": order,
            "quantity": quantity,
            "price": current_price,
            "side": side,
            "leverage": leverage,
            "type": order_params['type']
        }

    except Exception as e:
        print(f"❌ {symbol} {side} 订单执行失败: {e}")
        if logger:
            logger.error(f"{symbol} {side} 订单执行失败: {e}")
        return False, None


def set_dynamic_stop_loss_take_profit(client, symbol, side, entry_price, quantity, sl_pct=3.0, tp_pct=6.0, logger=None):
    """
    设置动态止盈止损

    参数:
        client: Binance客户端
        symbol: 交易对
        side: 交易方向 ('LONG' or 'SHORT')
        entry_price: 入场价格
        quantity: 交易数量
        sl_pct: 止损百分比
        tp_pct: 止盈百分比
        logger: 日志对象（可选）

    返回:
        success: 是否成功设置
        orders: 止盈止损订单信息
    """
    try:
        # 计算止盈止损价格
        if side == "LONG":
            sl_price = entry_price * (1 - sl_pct / 100)
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_side = "SELL"
            tp_side = "SELL"
        else:  # SHORT
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_side = "BUY"
            tp_side = "BUY"

        print(f"🔄 {symbol} 设置动态止盈止损 - 方向: {side}, 入场价: {entry_price:.6f}")
        print(f"⛔ 止损价: {sl_price:.6f} ({sl_pct}%), 🎯 止盈价: {tp_price:.6f} ({tp_pct}%)")

        # 设置止损订单
        stop_loss_order = client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type="STOP_MARKET",
            quantity=str(quantity),
            stopPrice=str(round(sl_price, 6)),
            reduceOnly=True,
            positionSide=side
        )

        # 设置止盈订单
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type="TAKE_PROFIT_MARKET",
            quantity=str(quantity),
            stopPrice=str(round(tp_price, 6)),
            reduceOnly=True,
            positionSide=side
        )

        if logger:
            logger.info(f"{symbol} {side} 止盈止损设置成功", extra={
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct
            })

        print(f"✅ {symbol} 止盈止损设置成功")

        return True, {
            "stop_loss": stop_loss_order,
            "take_profit": take_profit_order,
            "sl_price": sl_price,
            "tp_price": tp_price
        }

    except Exception as e:
        print(f"❌ {symbol} 设置止盈止损失败: {e}")
        if logger:
            logger.error(f"{symbol} 设置止盈止损失败: {e}")
        return False, None


def calculate_trailing_stop_params(quality_score, trend, market_conditions):
    """
    根据质量评分和市场情况计算适合的移动止损参数

    参数:
        quality_score: 质量评分 (0-10)
        trend: 市场趋势 ("UP", "DOWN", "NEUTRAL")
        market_conditions: 市场环境信息

    返回:
        activation_pct: 移动止损激活百分比
        callback_pct: 回调百分比
    """
    # 基础激活百分比
    if quality_score >= 8.0:
        activation_pct = 2.0  # 高质量信号，快速激活移动止损
    elif quality_score >= 6.0:
        activation_pct = 3.0  # 中等质量信号
    else:
        activation_pct = 4.0  # 较低质量信号，需要更多确认

    # 基础回调百分比
    if quality_score >= 8.0:
        callback_pct = 1.0  # 高质量信号，紧密跟踪
    elif quality_score >= 6.0:
        callback_pct = 1.5  # 中等质量信号
    else:
        callback_pct = 2.0  # 较低质量信号，更宽松的跟踪

    # 根据趋势调整
    if trend == "UP" or trend == "DOWN":
        # 明确趋势，可以更紧密地跟踪
        callback_pct *= 0.8
    else:
        # 中性趋势，需要更宽松的跟踪
        callback_pct *= 1.2
        activation_pct *= 1.2

    # 根据市场条件调整
    if market_conditions and 'environment' in market_conditions:
        env = market_conditions['environment']
        if env == 'trending':
            # 趋势市场，可以更紧密地跟踪
            callback_pct *= 0.8
        elif env == 'ranging':
            # 震荡市场，需要更宽松的跟踪
            callback_pct *= 1.5
            activation_pct *= 1.3
        elif env == 'breakout':
            # 突破市场，快速激活但宽松跟踪
            activation_pct *= 0.7
            callback_pct *= 1.2
        elif env == 'extreme_volatility':
            # 极端波动市场，非常宽松的跟踪
            callback_pct *= 2.0
            activation_pct *= 1.5

    # 确保值在合理范围内
    activation_pct = max(1.0, min(10.0, activation_pct))
    callback_pct = max(0.5, min(5.0, callback_pct))

    print(f"🔄 移动止损参数 - 激活: {activation_pct:.1f}%, 回调: {callback_pct:.1f}%")
    return activation_pct, callback_pct