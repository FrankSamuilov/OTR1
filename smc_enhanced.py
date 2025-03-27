import numpy as np
import pandas as pd
from data_module import get_historical_data
from indicators_module import calculate_optimized_indicators, get_smc_trend_and_duration, find_swing_points, \
    calculate_fibonacci_retracements


def calculate_quality_score(df, client=None, symbol=None, btc_df=None, config=None, logger=None):
    """
    计算0-10分的货币质量评分，10分表示低风险
    基于SMC策略（Smart Money Concept）和风险参数

    参数:
        df (DataFrame): 包含价格和计算指标的数据
        client: Binance客户端（可选）
        symbol: 交易对符号（可选）
        btc_df: BTC数据（可选，用于市场情绪评估）
        config: 配置对象（可选）
        logger: 日志对象（可选）

    返回:
        quality_score (float): 0-10分的质量评分
        metrics (dict): 计算过程中的指标明细
    """
    # 创建指标字典用于返回
    metrics = {}

    # 防御性检查
    if df is None or len(df) < 20:
        if logger:
            logger.warning(f"{symbol}数据不足，无法计算质量评分")
        print(f"⚠️ {symbol}数据不足，无法计算质量评分")
        return 0.0, {'error': 'insufficient_data'}

    # 基本风险评估 (3分)
    risk_score = 3.0
    print(f"📊 {symbol} - 基础风险评分: {risk_score}")

    # 1. 市场结构评估 (SMC核心) - 最高2分
    trend, duration, trend_info = get_smc_trend_and_duration(df, config, logger)
    metrics['trend'] = trend
    metrics['duration'] = duration
    print(f"📈 {symbol} - 市场趋势: {trend}, 持续时间: {duration}分钟")

    # 稳定上升趋势得高分
    if trend == "UP" and duration > 30:  # 缩短时间要求，从60分钟减至30分钟
        structure_score = 2.0
        print(f"✅ {symbol} - 稳定上升趋势，结构评分: 2.0")
    elif trend == "UP":
        structure_score = 1.5
        print(f"✅ {symbol} - 上升趋势，结构评分: 1.5")
    elif trend == "NEUTRAL":
        structure_score = 1.0
        print(f"⚖️ {symbol} - 中性趋势，结构评分: 1.0")
    elif trend == "DOWN" and duration > 30:  # 缩短时间要求
        structure_score = 0.5  # 风险较高
        print(f"⚠️ {symbol} - 明显下降趋势，结构评分: 0.5")
    else:
        structure_score = 0.8
        print(f"⚠️ {symbol} - 不明确趋势，结构评分: 0.8")
    metrics['structure_score'] = structure_score

    # 2. 订单块和流动性评估 - 最高2分
    try:
        # 成交量评估
        volume_mean = df['volume'].rolling(20).mean().iloc[-1]
        recent_volume = df['volume'].iloc[-1]
        volume_ratio = recent_volume / volume_mean if volume_mean > 0 else 1.0
        print(f"📊 {symbol} - 成交量比率: {volume_ratio:.2f}")

        # OBV趋势评估
        obv_trend = df['OBV'].iloc[-1] > df['OBV'].iloc[-5] if 'OBV' in df.columns and len(df) >= 5 else False
        print(f"📊 {symbol} - OBV趋势{'上升' if obv_trend else '下降'}")

        # ATR评估 - 波动率
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        atr_mean = df['ATR'].rolling(20).mean().iloc[-1] if 'ATR' in df.columns else 1
        atr_ratio = atr / atr_mean if atr_mean > 0 else 1.0
        print(f"📊 {symbol} - 波动率比率: {atr_ratio:.2f}")

        # 超级趋势评估
        supertrend_aligned = False
        if 'Supertrend_Direction' in df.columns:
            st_direction = df['Supertrend_Direction'].iloc[-1]
            supertrend_aligned = (st_direction > 0 and trend == "UP") or (st_direction < 0 and trend == "DOWN")
            print(
                f"📊 {symbol} - 超级趋势方向: {'上升' if st_direction > 0 else '下降'}, 与趋势一致: {supertrend_aligned}")

        # 订单块评估
        has_order_block = (volume_ratio > 1.3 and
                           abs(df['close'].iloc[-1] - df['close'].iloc[-2]) < atr)
        print(f"📊 {symbol} - 订单块检测: {'有' if has_order_block else '无'}")

        metrics['volume_ratio'] = volume_ratio
        metrics['atr_ratio'] = atr_ratio
        metrics['has_order_block'] = has_order_block
        metrics['supertrend_aligned'] = supertrend_aligned

        # 订单块评分
        if has_order_block and obv_trend and supertrend_aligned:
            order_block_score = 2.0
            print(f"✅ {symbol} - 订单块+OBV+超级趋势完美匹配，评分: 2.0")
        elif has_order_block and (obv_trend or supertrend_aligned):
            order_block_score = 1.5
            print(f"✅ {symbol} - 订单块部分匹配，评分: 1.5")
        elif has_order_block or obv_trend:
            order_block_score = 1.0
            print(f"⚖️ {symbol} - 有订单块或OBV趋势，评分: 1.0")
        elif volume_ratio > 0.8:
            order_block_score = 0.7
            print(f"⚠️ {symbol} - 成交量尚可，评分: 0.7")
        else:
            order_block_score = 0.5
            print(f"⚠️ {symbol} - 订单块评估不佳，评分: 0.5")

        # 波动性降分
        if atr_ratio > 1.5:  # 波动性高于平均的50%
            order_block_score *= 0.7  # 降低30%的评分
            print(f"⚠️ {symbol} - 波动性过高，订单块评分降至: {order_block_score:.2f}")

        metrics['order_block_score'] = order_block_score
    except Exception as e:
        if logger:
            logger.error(f"{symbol}订单块评估出错: {e}")
        order_block_score = 0.5
        metrics['order_block_error'] = str(e)
        print(f"❌ {symbol} - 订单块评估出错: {e}")

    # 3. 支撑阻力评估 - 最高2分
    try:
        swing_highs, swing_lows = find_swing_points(df)
        fib_levels = calculate_fibonacci_retracements(df)

        print(f"📊 {symbol} - 发现摆动高点: {len(swing_highs)}个, 摆动低点: {len(swing_lows)}个")
        if fib_levels:
            print(f"📊 {symbol} - 斐波那契水平: {[round(level, 4) for level in fib_levels[:3]]}...")

        current_price = df['close'].iloc[-1]

        # 确定当前支撑位和阻力位
        if len(swing_lows) >= 2:
            current_support = min(swing_lows[-1], swing_lows[-2])
        else:
            current_support = df['low'].min()

        if len(swing_highs) >= 2:
            current_resistance = max(swing_highs[-1], swing_highs[-2])
        else:
            current_resistance = df['high'].max()

        # 计算价格与支撑/阻力的距离
        support_distance = (current_price - current_support) / current_price
        resistance_distance = (current_resistance - current_price) / current_price

        print(f"📊 {symbol} - 距离支撑位: {support_distance:.2%}, 距离阻力位: {resistance_distance:.2%}")

        # 检查价格与斐波那契回撤位的位置
        near_fib_support = False
        fib_support_level = 0

        if fib_levels and len(fib_levels) >= 3:  # 确保有足够的斐波那契水平
            # 检查价格是否接近任何斐波那契支撑位
            for i, level in enumerate(fib_levels):
                if abs(current_price - level) / current_price < 0.01:  # 1%以内视为接近
                    near_fib_support = True
                    fib_support_level = i
                    print(f"✅ {symbol} - 价格接近斐波那契水平 {i}: {level:.4f}")
                    break

        metrics['support_distance'] = support_distance
        metrics['resistance_distance'] = resistance_distance
        metrics['near_fib_support'] = near_fib_support
        metrics['fib_support_level'] = fib_support_level

        # 支撑阻力评分
        if near_fib_support:
            # 黄金分割较高位置得分更高
            sr_score = 2.0 - (fib_support_level * 0.3)  # 0.382得2.0分，0.618得1.7分
            print(f"✅ {symbol} - 价格位于黄金分割位置，支撑阻力评分: {sr_score:.2f}")
        elif support_distance < 0.01 and resistance_distance > 0.05:
            # 接近支撑且远离阻力
            sr_score = 1.8
            print(f"✅ {symbol} - 价格接近支撑且远离阻力，支撑阻力评分: 1.8")
        elif support_distance < 0.03:
            # 相对接近支撑
            sr_score = 1.5
            print(f"✅ {symbol} - 价格相对接近支撑，支撑阻力评分: 1.5")
        elif resistance_distance < 0.03:
            # 相对接近阻力
            sr_score = 0.8
            print(f"⚠️ {symbol} - 价格接近阻力，支撑阻力评分: 0.8")
        else:
            # 处于中间位置
            sr_score = 1.0
            print(f"⚖️ {symbol} - 价格处于中间位置，支撑阻力评分: 1.0")

        metrics['sr_score'] = sr_score
    except Exception as e:
        if logger:
            logger.error(f"{symbol}支撑阻力评估出错: {e}")
        sr_score = 1.0
        metrics['sr_error'] = str(e)
        print(f"❌ {symbol} - 支撑阻力评估出错: {e}")

    # 4. 技术指标评估 - 最高2分
    try:
        # MACD
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        macd_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else 0
        macd_cross = macd > macd_signal
        print(f"📊 {symbol} - MACD: {macd:.6f}, Signal: {macd_signal:.6f}, 金叉: {macd_cross}")

        # RSI
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        rsi_healthy = 30 <= rsi <= 70
        print(f"📊 {symbol} - RSI: {rsi:.2f}, 健康区间: {rsi_healthy}")

        # 均线
        ema5 = df['EMA5'].iloc[-1] if 'EMA5' in df.columns else 0
        ema20 = df['EMA20'].iloc[-1] if 'EMA20' in df.columns else 0
        price_above_ema = df['close'].iloc[-1] > ema20
        print(f"📊 {symbol} - EMA5: {ema5:.4f}, EMA20: {ema20:.4f}, 价格高于EMA20: {price_above_ema}")

        # 布林带
        bb_width = (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]) / df['BB_Middle'].iloc[-1] if all(
            x in df.columns for x in ['BB_Upper', 'BB_Lower', 'BB_Middle']) else 0.1
        print(f"📊 {symbol} - 布林带宽度: {bb_width:.4f}")

        metrics['macd_cross'] = macd_cross
        metrics['rsi'] = rsi
        metrics['rsi_healthy'] = rsi_healthy
        metrics['price_above_ema'] = price_above_ema
        metrics['bb_width'] = bb_width

        # 技术指标评分
        tech_score = 0.0

        # MACD交叉向上且RSI健康 +1.0
        if macd_cross and rsi_healthy:
            tech_score += 1.0
            print(f"✅ {symbol} - MACD金叉+RSI健康，技术加分: +1.0")
        # RSI健康但无交叉 +0.6
        elif rsi_healthy:
            tech_score += 0.6
            print(f"✅ {symbol} - RSI健康，技术加分: +0.6")
        # RSI超买或超卖 -0.2
        else:
            tech_score -= 0.2
            print(f"⚠️ {symbol} - RSI超买或超卖，技术减分: -0.2")

        # 价格在均线上方 +0.5
        if price_above_ema:
            tech_score += 0.5
            print(f"✅ {symbol} - 价格在均线上方，技术加分: +0.5")

        # 考虑布林带宽度 (标准情况下分值0.5，宽度越小越好)
        if bb_width < 0.03:  # 非常紧缩，可能即将突破
            tech_score += 0.5
            print(f"✅ {symbol} - 布林带紧缩，技术加分: +0.5")
        elif bb_width < 0.06:  # 较紧缩
            tech_score += 0.3
            print(f"✅ {symbol} - 布林带较紧缩，技术加分: +0.3")
        elif bb_width > 0.08:  # 较宽，波动较大
            tech_score -= 0.2
            print(f"⚠️ {symbol} - 布林带过宽，技术减分: -0.2")

        # 确保在范围内
        tech_score = max(0.0, min(2.0, tech_score))
        print(f"📊 {symbol} - 最终技术指标评分: {tech_score:.2f}")
        metrics['tech_score'] = tech_score
    except Exception as e:
        if logger:
            logger.error(f"{symbol}技术指标评估出错: {e}")
        tech_score = 0.8
        metrics['tech_error'] = str(e)
        print(f"❌ {symbol} - 技术指标评估出错: {e}")

    # 5. 市场情绪评估 - 最高1分
    try:
        market_score = 0.5  # 默认中性
        print(f"📊 {symbol} - 默认市场情绪评分: 0.5")

        # 如果提供了BTC数据，评估整体市场情绪
        if btc_df is not None and len(btc_df) > 5:
            btc_change = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-5]) / btc_df['close'].iloc[-5]
            print(f"📊 {symbol} - BTC变化率: {btc_change:.2%}")

            if btc_change > 0.02:  # BTC上涨超过2%
                market_score = 1.0
                print(f"✅ {symbol} - BTC强势上涨，市场情绪评分: 1.0")
            elif btc_change > 0.005:  # BTC小幅上涨
                market_score = 0.8
                print(f"✅ {symbol} - BTC小幅上涨，市场情绪评分: 0.8")
            elif btc_change < -0.02:  # BTC下跌超过2%
                market_score = 0.2
                print(f"⚠️ {symbol} - BTC强势下跌，市场情绪评分: 0.2")
            elif btc_change < -0.005:  # BTC小幅下跌
                market_score = 0.3
                print(f"⚠️ {symbol} - BTC小幅下跌，市场情绪评分: 0.3")

        # 如果提供了客户端和符号，也可以查看期货资金费率
        if client and symbol:
            try:
                funding_rate = float(client.futures_mark_price(symbol=symbol)['lastFundingRate'])
                print(f"📊 {symbol} - 资金费率: {funding_rate:.6f}")

                # 负的资金费率通常对做多有利
                if funding_rate < -0.0002:  # 明显为负
                    market_score += 0.1
                    print(f"✅ {symbol} - 负资金费率，市场情绪加分: +0.1")
                elif funding_rate > 0.0002:  # 明显为正
                    market_score -= 0.1
                    print(f"⚠️ {symbol} - 正资金费率，市场情绪减分: -0.1")
            except Exception as e:
                print(f"⚠️ {symbol} - 无法获取资金费率: {e}")
                pass  # 忽略资金费率获取错误

        metrics['market_score'] = market_score
    except Exception as e:
        if logger:
            logger.error(f"{symbol}市场情绪评估出错: {e}")
        market_score = 0.5
        metrics['market_error'] = str(e)
        print(f"❌ {symbol} - 市场情绪评估出错: {e}")

    # 震荡市场检测与降分
    is_ranging = False
    if 'ADX' in df.columns:
        adx = df['ADX'].iloc[-1]
        if adx < 20:
            is_ranging = True
            # 震荡市场降分
            quality_penalty = 2.0  # 在震荡市场降低2分
            print(f"⚠️ {symbol} - 检测到震荡市场 (ADX: {adx:.2f} < 20)，评分惩罚: -2.0")
            metrics['is_ranging'] = True
            metrics['adx_value'] = adx
        else:
            print(f"📊 {symbol} - ADX: {adx:.2f} >= 20，非震荡市场")

    # 汇总得分
    quality_score = risk_score + structure_score + order_block_score + sr_score + tech_score + market_score

    # 对震荡市场进行惩罚
    if is_ranging:
        quality_score = max(0.0, quality_score - 2.0)

    # 对趋势不明确的市场降分
    if trend == "NEUTRAL":
        quality_score *= 0.8
        print(f"⚠️ {symbol} - 趋势不明确，总评分乘以0.8")

    # 确保最终分数在0-10范围内
    quality_score = max(0.0, min(10.0, quality_score))

    # 记录所有评分组成
    metrics['risk_score'] = risk_score
    metrics['final_score'] = quality_score

    print(f"🏆 {symbol} - 最终质量评分: {quality_score:.2f}")
    print(
        f"组成: 风险({risk_score}) + 结构({structure_score}) + 订单块({order_block_score}) + 支撑阻力({sr_score}) + 技术({tech_score}) + 市场({market_score})")

    if logger:
        logger.info(f"{symbol}质量评分: {quality_score:.2f}", extra=metrics)

    return quality_score, metrics


def detect_pattern_similarity(df, historical_dfs, window_length=10, similarity_threshold=0.8, logger=None):
    """
    检测当前市场模式与历史模式的相似度，仅专注于上涨与下跌模式

    参数:
        df (DataFrame): 当前市场数据
        historical_dfs (list): 历史数据框列表，每个元素都是包含时间戳的DataFrame
        window_length (int): 比较窗口长度，默认10
        similarity_threshold (float): 相似度阈值
        logger: 日志对象（可选）

    返回:
        similarity_info (dict): 包含最高相似度和相应时间的信息
    """
    if df is None or len(df) < window_length:
        print("数据长度不足，无法检测模式相似性")
        return {'max_similarity': 0, 'similar_time': None, 'is_similar': False, 'pattern_type': None}

    # 提取当前模式特征
    try:
        # 使用价格变化率作为特征，减少绝对价格的影响
        current_pattern = []
        pattern_direction = 0  # 0=中性, 1=上涨, -1=下跌

        # 确定当前模式方向
        first_price = df['close'].iloc[-window_length]
        last_price = df['close'].iloc[-1]
        overall_change = (last_price - first_price) / first_price

        if overall_change > 0.01:  # 上涨超过1%
            pattern_direction = 1
            print(f"当前价格模式：上涨 (+{overall_change:.2%})")
        elif overall_change < -0.01:  # 下跌超过1%
            pattern_direction = -1
            print(f"当前价格模式：下跌 ({overall_change:.2%})")
        else:
            print(f"当前价格模式：盘整 ({overall_change:.2%})")

        # 提取每日变化率
        for i in range(1, window_length):
            # 使用收盘价变化率
            change_rate = df['close'].iloc[-i] / df['close'].iloc[-i - 1] - 1
            current_pattern.append(change_rate)

        current_pattern = np.array(current_pattern)
        print(f"当前模式变化率（最近{len(current_pattern)}个周期）：{[f'{x:.2%}' for x in current_pattern[:3]]}...")

        # 寻找最相似的历史模式
        max_similarity = 0
        similar_time = None
        similar_direction = None

        for hist_df in historical_dfs:
            if hist_df is None or len(hist_df) < window_length + 1:
                continue

            # 对每个可能的窗口计算相似度
            for i in range(len(hist_df) - window_length):
                # 确定历史模式方向
                hist_first_price = hist_df['close'].iloc[i]
                hist_last_price = hist_df['close'].iloc[i + window_length - 1]
                hist_overall_change = (hist_last_price - hist_first_price) / hist_first_price

                hist_direction = 0
                if hist_overall_change > 0.01:
                    hist_direction = 1
                elif hist_overall_change < -0.01:
                    hist_direction = -1

                # 如果只关注相同方向的模式，跳过不同方向的
                if pattern_direction != 0 and hist_direction != 0 and pattern_direction != hist_direction:
                    continue

                # 计算历史模式
                hist_pattern = []
                for j in range(1, window_length):
                    hist_change_rate = hist_df['close'].iloc[i + j] / hist_df['close'].iloc[i + j - 1] - 1
                    hist_pattern.append(hist_change_rate)

                hist_pattern = np.array(hist_pattern)

                # 计算欧几里得距离
                if len(hist_pattern) == len(current_pattern):
                    distance = np.sqrt(np.sum((current_pattern - hist_pattern) ** 2))
                    # 最大可能距离（假设每个点变化率相差2, 即一个+100%一个-100%）
                    max_distance = np.sqrt(window_length * 4)
                    # 归一化为相似度
                    similarity = 1 - (distance / max_distance)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        # 获取这个窗口的时间
                        if 'time' in hist_df.columns:
                            similar_time = hist_df['time'].iloc[i]
                        else:
                            similar_time = f"索引位置 {i}"
                        similar_direction = hist_direction

        # 是否达到相似度阈值
        is_similar = max_similarity >= similarity_threshold

        pattern_type = None
        if similar_direction == 1:
            pattern_type = "uptrend"
        elif similar_direction == -1:
            pattern_type = "downtrend"

        if is_similar:
            print(f"检测到高相似度模式: {max_similarity:.2%}, 时间: {similar_time}, 类型: {pattern_type or '中性'}")
        else:
            print(f"未检测到高相似度模式，最高相似度: {max_similarity:.2%}")

        similarity_info = {
            'max_similarity': max_similarity,
            'similar_time': similar_time,
            'is_similar': is_similar,
            'pattern_type': pattern_type,
            'current_direction': pattern_direction
        }

        return similarity_info
    except Exception as e:
        if logger:
            logger.error(f"模式相似度检测出错: {e}")
        print(f"模式相似度检测出错: {e}")
        return {'max_similarity': 0, 'similar_time': None, 'is_similar': False, 'pattern_type': None, 'error': str(e)}


def adjust_quality_for_similarity(quality_score, similarity_info, adjustment_factor=0.2):
    """
    根据相似度信息调整质量评分，加大调整因子

    参数:
        quality_score (float): 初始质量评分
        similarity_info (dict): 相似度信息
        adjustment_factor (float): 调整因子，默认0.2（20%）

    返回:
        adjusted_score (float): 调整后的评分
    """
    if not similarity_info['is_similar']:
        print("无相似模式，保持原始评分")
        return quality_score

    # 对于高相似度，根据历史模式表现调整评分
    similarity = similarity_info['max_similarity']
    pattern_type = similarity_info['pattern_type']

    # 基于模式类型调整
    if pattern_type == "uptrend":
        # 上升趋势相似模式，略微提高评分
        adjustment = quality_score * adjustment_factor * (similarity - 0.8) / 0.2
        print(f"检测到上升趋势相似模式，评分调整: +{adjustment:.2f}")
    elif pattern_type == "downtrend":
        # 下降趋势相似模式，略微降低评分
        adjustment = -quality