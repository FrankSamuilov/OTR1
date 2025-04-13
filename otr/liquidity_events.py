"""
流动性事件检测模块
用于检测和分析市场中的流动性事件，包括:
1. 高点或低点流动性吸收
2. "止损猎杀"后的反转
3. 价格不平衡区域形成
4. 机构交易意图显示
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from logger_utils import Colors, print_colored
from indicators_module import find_swing_points, calculate_fibonacci_retracements, get_smc_trend_and_duration


def detect_liquidity_events(df, current_price=None):
    """
    检测并分析价格图表中的流动性事件

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格(如果为None则使用最新收盘价)

    返回:
        dict: 包含已识别流动性事件的分析结果
    """
    if current_price is None:
        current_price = df['close'].iloc[-1]

    print_colored(f"开始检测流动性事件，当前价格: {current_price:.6f}", Colors.BLUE)

    # 初始化结果字典
    results = {
        "detected_events": [],
        "high_value_zones": [],
        "quality_score": 0,
        "liquidity_absorption": {
            "detected": False,
            "details": {}
        },
        "stop_hunt": {
            "detected": False,
            "details": {}
        },
        "imbalance": {
            "detected": False,
            "zones": []
        },
        "institutional_intent": {
            "detected": False,
            "evidence": []
        }
    }

    # 检测各种流动性事件
    liquidity_absorption = detect_liquidity_absorption(df, current_price)
    stop_hunt = detect_stop_hunt(df, current_price)
    imbalance = detect_price_imbalance(df, current_price)
    inst_intent = detect_institutional_intent(df, current_price)

    # 更新结果
    results["liquidity_absorption"] = liquidity_absorption
    results["stop_hunt"] = stop_hunt
    results["imbalance"] = imbalance
    results["institutional_intent"] = inst_intent

    # 汇总检测到的事件
    if liquidity_absorption["detected"]:
        results["detected_events"].append({
            "type": "liquidity_absorption",
            "level": liquidity_absorption["level"],
            "strength": liquidity_absorption["strength"],
            "direction": liquidity_absorption["direction"]
        })

    if stop_hunt["detected"]:
        results["detected_events"].append({
            "type": "stop_hunt",
            "level": stop_hunt["level"],
            "direction": stop_hunt["direction"],
            "strength": stop_hunt["strength"]
        })

    if imbalance["detected"] and imbalance["recent_zone"]:
        results["detected_events"].append({
            "type": "imbalance",
            "zone": imbalance["recent_zone"],
            "strength": min(10, imbalance["recent_zone"]["size"] * 10)
        })

    if inst_intent["detected"]:
        results["detected_events"].append({
            "type": "institutional_intent",
            "evidence_count": len(inst_intent["evidence"]),
            "strength": inst_intent["intent_score"]
        })

    # 计算综合质量评分
    quality_score = calculate_liquidity_event_quality(results["detected_events"])
    results["quality_score"] = quality_score

    # 输出检测结果
    if results["detected_events"]:
        print_colored(f"检测到 {len(results['detected_events'])} 个流动性事件，质量评分: {quality_score:.2f}/10",
                      Colors.GREEN if quality_score > 5 else Colors.YELLOW)

        for event in results["detected_events"]:
            event_type = event["type"]
            if event_type == "liquidity_absorption":
                direction = "上方" if event["direction"] == "high" else "下方"
                print_colored(f"  - {direction}流动性吸收，等级: {event['level']:.6f}, 强度: {event['strength']:.2f}",
                              Colors.GREEN)
            elif event_type == "stop_hunt":
                direction = "向下突破后反转" if event["direction"] == "down_up" else "向上突破后反转"
                print_colored(f"  - 止损猎杀，{direction}，等级: {event['level']:.6f}, 强度: {event['strength']:.2f}",
                              Colors.RED)
            elif event_type == "imbalance":
                zone = event["zone"]
                zone_type = "买方" if zone["type"] == "bullish" else "卖方"
                print_colored(f"  - 价格不平衡区域，{zone_type}，范围: {zone['start_price']:.6f}-{zone['end_price']:.6f}",
                              Colors.YELLOW)
            elif event_type == "institutional_intent":
                print_colored(f"  - 机构交易意图，证据数量: {event['evidence_count']}, 强度: {event['strength']:.2f}",
                              Colors.BLUE)
    else:
        print_colored("未检测到明显的流动性事件", Colors.YELLOW)

    return results


def detect_liquidity_absorption(df, current_price):
    """
    检测高点或低点的流动性吸收

    特征:
    - 价格突破先前的摆动高点/低点
    - 突破后快速反转
    - 通常伴随较大成交量
    """
    result = {
        "detected": False,
        "level": None,
        "strength": 0,
        "direction": None
    }

    # 获取摆动点
    swing_highs, swing_lows = find_swing_points(df)
    if not swing_highs or not swing_lows:
        return result

    # 最近的N根K线
    lookback = min(10, len(df) - 1)
    recent_data = df.iloc[-lookback:]

    # 检测高点流动性吸收
    recent_high = recent_data['high'].max()
    recent_high_idx = recent_data['high'].idxmax()

    # 检查是否突破了之前的摆动高点
    for high in sorted(swing_highs):
        # 如果最近的高点突破了摆动高点但随后回落
        if recent_high > high and current_price < high * 1.01:  # 允许1%的回落容差
            # 检查成交量确认
            if 'volume' in recent_data.columns:
                vol_increase = recent_data.loc[recent_high_idx, 'volume'] / recent_data['volume'].mean()
                # 如果成交量显著增加
                if vol_increase > 1.5:
                    result["detected"] = True
                    result["level"] = high
                    result["strength"] = min(10, 5 + (vol_increase - 1.5) * 2)
                    result["direction"] = "high"
                    return result

    # 检测低点流动性吸收
    recent_low = recent_data['low'].min()
    recent_low_idx = recent_data['low'].idxmin()

    # 检查是否突破了之前的摆动低点
    for low in sorted(swing_lows, reverse=True):
        # 如果最近的低点突破了摆动低点但随后反弹
        if recent_low < low and current_price > low * 0.99:  # 允许1%的反弹容差
            # 检查成交量确认
            if 'volume' in recent_data.columns:
                vol_increase = recent_data.loc[recent_low_idx, 'volume'] / recent_data['volume'].mean()
                # 如果成交量显著增加
                if vol_increase > 1.5:
                    result["detected"] = True
                    result["level"] = low
                    result["strength"] = min(10, 5 + (vol_increase - 1.5) * 2)
                    result["direction"] = "low"
                    return result

    return result


def detect_stop_hunt(df, current_price):
    """
    检测止损猎杀行为

    特征:
    - 价格突破重要支撑/阻力位后快速反转
    - 通常快速且幅度巨大
    - 伴随异常的成交量
    """
    result = {
        "detected": False,
        "level": None,
        "direction": None,
        "strength": 0
    }

    # 获取最近的N根K线
    lookback = min(20, len(df) - 1)
    recent_data = df.iloc[-lookback:]

    # 检查价格突破关键水平后的反转

    # 1. 确定潜在的支撑位和阻力位
    supports_resistances = []

    # 使用常见技术指标作为支撑/阻力位
    for indicator in ['Supertrend', 'EMA20', 'EMA50', 'SMA50', 'BB_Lower', 'BB_Upper']:
        if indicator in df.columns:
            supports_resistances.append(df[indicator].iloc[-lookback])

    # 如果找不到指标，使用摆动点
    if not supports_resistances:
        swing_highs, swing_lows = find_swing_points(df)
        if swing_highs:
            supports_resistances.extend(swing_highs[-3:])
        if swing_lows:
            supports_resistances.extend(swing_lows[-3:])

    # 2. 检查每个支撑/阻力位是否发生了止损猎杀
    for level in supports_resistances:
        # 向下突破后反弹 (上方止损被猎杀)
        if any(recent_data['low'] < level) and current_price > level * 0.995:
            # 计算突破的最大深度
            min_price = recent_data['low'].min()
            max_depth = (level - min_price) / level * 100  # 百分比

            # 检查成交量确认
            if 'volume' in recent_data.columns:
                vol_ratio = recent_data['volume'].max() / recent_data['volume'].mean()

                # 如果深度和成交量都符合条件
                if max_depth > 0.5 and vol_ratio > 1.8:
                    result["detected"] = True
                    result["level"] = level
                    result["direction"] = "down_up"  # 向下突破后向上反转
                    result["strength"] = min(10, max_depth * 2)
                    return result

        # 向上突破后回落 (下方止损被猎杀)
        if any(recent_data['high'] > level) and current_price < level * 1.005:
            # 计算突破的最大高度
            max_price = recent_data['high'].max()
            max_height = (max_price - level) / level * 100  # 百分比

            # 检查成交量确认
            if 'volume' in recent_data.columns:
                vol_ratio = recent_data['volume'].max() / recent_data['volume'].mean()

                # 如果高度和成交量都符合条件
                if max_height > 0.5 and vol_ratio > 1.8:
                    result["detected"] = True
                    result["level"] = level
                    result["direction"] = "up_down"  # 向上突破后向下反转
                    result["strength"] = min(10, max_height * 2)
                    return result

    return result


def detect_price_imbalance(df, current_price):
    """
    检测价格不平衡区域

    特征:
    - 价格在短时间内快速跳跃，留下"缺口"
    - 通常伴随异常成交量
    - 不平衡区域往往被填补
    """
    result = {
        "detected": False,
        "zones": [],
        "recent_zone": None
    }

    # 最小不平衡区域大小 (相对于ATR的倍数)
    min_gap_size = 0.5

    # 获取ATR
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
    else:
        # 简单估计ATR
        atr = (df['high'] - df['low']).mean()

    # 寻找不平衡区域
    imbalance_zones = []

    # 检查相邻蜡烛之间的缺口
    for i in range(1, min(30, len(df) - 1)):
        prev_idx = len(df) - i - 1
        curr_idx = len(df) - i

        prev_candle = df.iloc[prev_idx]
        curr_candle = df.iloc[curr_idx]

        # 向上缺口
        if prev_candle['high'] < curr_candle['low']:
            gap_size = curr_candle['low'] - prev_candle['high']

            # 如果缺口足够大
            if gap_size > atr * min_gap_size:
                zone = {
                    "type": "bullish",
                    "start_price": prev_candle['high'],
                    "end_price": curr_candle['low'],
                    "size": gap_size,
                    "index": curr_idx,
                    "filled": current_price <= prev_candle['high']
                }
                imbalance_zones.append(zone)

        # 向下缺口
        elif prev_candle['low'] > curr_candle['high']:
            gap_size = prev_candle['low'] - curr_candle['high']

            # 如果缺口足够大
            if gap_size > atr * min_gap_size:
                zone = {
                    "type": "bearish",
                    "start_price": prev_candle['low'],
                    "end_price": curr_candle['high'],
                    "size": gap_size,
                    "index": curr_idx,
                    "filled": current_price >= prev_candle['low']
                }
                imbalance_zones.append(zone)

    # 如果找到了不平衡区域
    if imbalance_zones:
        result["detected"] = True
        result["zones"] = imbalance_zones

        # 按时间顺序（从最近到最远）排序
        imbalance_zones.sort(key=lambda x: x["index"], reverse=True)

        # 找出最近的未填补区域
        unfilled_zones = [z for z in imbalance_zones if not z["filled"]]
        if unfilled_zones:
            result["recent_zone"] = unfilled_zones[0]
        else:
            # 如果所有区域都已填补，选择最近的
            result["recent_zone"] = imbalance_zones[0]

    return result


def detect_institutional_intent(df, current_price):
    """
    检测机构交易意图的迹象

    特征:
    - 关键水平的异常成交量
    - 价格多次测试但未突破的水平
    - 对称的价格模式后跟随方向明确的突破
    """
    result = {
        "detected": False,
        "evidence": [],
        "intent_score": 0
    }

    # 机构意图证据
    evidence = []

    # 1. 检查异常成交量模式
    if 'volume' in df.columns:
        # 最近的N根K线
        lookback = min(30, len(df) - 1)
        recent_data = df.iloc[-lookback:]

        # 计算平均成交量和标准差
        mean_vol = recent_data['volume'].mean()
        std_vol = recent_data['volume'].std()

        # 检查是否有成交量特别大的K线
        volume_spikes = recent_data[recent_data['volume'] > mean_vol + 2 * std_vol]

        if not volume_spikes.empty:
            # 分析成交量尖峰是在关键价格水平还是随机出现
            for idx in volume_spikes.index:
                spike_candle = df.loc[idx]

                # 检查是否在关键价格水平
                is_at_key_level = False

                # 检查是否在移动平均线附近
                for ma in ['EMA20', 'EMA50', 'SMA50', 'SMA200']:
                    if ma in df.columns:
                        ma_value = df.loc[idx, ma]
                        # 如果价格在均线附近
                        if abs(spike_candle['close'] - ma_value) / ma_value < 0.01:
                            is_at_key_level = True
                            break

                if is_at_key_level:
                    evidence.append({
                        "type": "volume_spike_at_key_level",
                        "index": idx,
                        "volume_ratio": spike_candle['volume'] / mean_vol,
                        "price": spike_candle['close']
                    })

    # 2. 检查价格水平多次测试但未突破
    swing_highs, swing_lows = find_swing_points(df)

    # 聚类相近的摆动点，找出反复测试的水平
    if swing_highs:
        clusters = cluster_price_levels(swing_highs, threshold=0.005)
        for cluster in clusters:
            if len(cluster) >= 3:  # 如果一个水平被测试了至少3次
                evidence.append({
                    "type": "multiple_tests_of_resistance",
                    "level": np.mean(cluster),
                    "test_count": len(cluster)
                })

    if swing_lows:
        clusters = cluster_price_levels(swing_lows, threshold=0.005)
        for cluster in clusters:
            if len(cluster) >= 3:  # 如果一个水平被测试了至少3次
                evidence.append({
                    "type": "multiple_tests_of_support",
                    "level": np.mean(cluster),
                    "test_count": len(cluster)
                })

    # 3. 检查价格结构变化后的强方向性走势
    if len(df) >= 20:
        # 计算20根K线内的平均涨跌幅
        avg_change = abs(df['close'].pct_change().mean())

        # 最近5根K线的单向变化幅度
        recent_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]

        # 如果最近变化明显大于平均变化，可能是机构推动
        if abs(recent_change) > avg_change * 3:
            evidence.append({
                "type": "strong_directional_move",
                "change_pct": recent_change * 100,
                "direction": "up" if recent_change > 0 else "down"
            })

    # 更新结果
    if evidence:
        result["detected"] = True
        result["evidence"] = evidence

        # 计算意图得分
        intent_score = 0
        for item in evidence:
            if item["type"] == "volume_spike_at_key_level":
                intent_score += min(5, item["volume_ratio"])
            elif item["type"] in ["multiple_tests_of_resistance", "multiple_tests_of_support"]:
                intent_score += item["test_count"] * 1.5
            elif item["type"] == "strong_directional_move":
                intent_score += min(5, abs(item["change_pct"]) * 0.5)

        result["intent_score"] = min(10, intent_score)

    return result


def cluster_price_levels(price_levels, threshold=0.005):
    """聚类接近的价格水平"""
    if not price_levels:
        return []

    sorted_levels = sorted(price_levels)
    clusters = []
    current_cluster = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
            current_cluster.append(level)
        else:
            clusters.append(current_cluster)
            current_cluster = [level]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def calculate_liquidity_event_quality(events):
    """计算流动性事件的综合质量评分"""
    if not events:
        return 0

    total_score = 0

    for event in events:
        event_type = event["type"]

        if event_type == "liquidity_absorption":
            total_score += event["strength"] * 1.0  # 高权重
        elif event_type == "stop_hunt":
            total_score += event["strength"] * 0.8
        elif event_type == "imbalance":
            # 根据不平衡区域大小和是否被填补计算分数
            if "zone" in event:
                zone = event["zone"]
                score = min(5, event.get("strength", 0))
                total_score += score * (0.5 if zone.get("filled", False) else 1.0)
        elif event_type == "institutional_intent":
            total_score += min(8, event.get("strength", 0))

    # 限制最大分数为10
    return min(10, total_score)


def evaluate_price_position_with_liquidity(df, current_price=None):
    """
    综合评估价格位置条件和流动性事件

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格(如果为None则使用最新收盘价)

    返回:
        dict: 综合评估结果
    """
    from smc_entry_conditions import evaluate_price_position_conditions

    if current_price is None:
        current_price = df['close'].iloc[-1]

    # 获取原始价格位置评估
    print_colored("\n==== 原始价格位置评估 ====", Colors.BLUE)
    position_evaluation = evaluate_price_position_conditions(df, current_price)

    # 获取流动性事件评估
    print_colored("\n==== 流动性事件评估 ====", Colors.BLUE)
    liquidity_events = detect_liquidity_events(df, current_price)

    # 整合评估结果
    combined_evaluation = position_evaluation.copy()
    combined_evaluation["liquidity_events"] = liquidity_events

    # 调整综合得分
    original_score = position_evaluation["score"]
    liquidity_score = liquidity_events["quality_score"]

    # 流动性事件对总分的影响
    if liquidity_score > 7:
        # 高质量流动性事件强烈提升总分
        combined_score = min(10, original_score + (liquidity_score - 5) * 0.5)
    elif liquidity_score > 4:
        # 中等质量流动性事件适度提升总分
        combined_score = min(10, original_score + (liquidity_score - 4) * 0.3)
    else:
        # 低质量流动性事件略微提升总分
        combined_score = min(10, original_score + liquidity_score * 0.1)

    combined_evaluation["combined_score"] = combined_score

    # 调整最佳入场价格计算，考虑流动性事件
    if liquidity_events["detected_events"]:
        best_entry = calculate_optimal_entry_with_liquidity(
            df, current_price,
            position_evaluation["best_entry_price"],
            liquidity_events
        )
        combined_evaluation["best_entry_price"] = best_entry

    # 打印分析结果
    print_colored("\n==== 整合流动性事件的价格位置评估 ====", Colors.BLUE + Colors.BOLD)
    print_colored(f"原始价格位置评分: {original_score:.2f}/10", Colors.INFO)
    print_colored(f"流动性事件评分: {liquidity_score:.2f}/10", Colors.INFO)
    print_colored(f"综合评分: {combined_score:.2f}/10",
                  Colors.GREEN if combined_score >= 7 else
                  Colors.YELLOW if combined_score >= 5 else
                  Colors.RED)

    if liquidity_events["detected_events"]:
        print_colored(f"检测到的流动性事件:", Colors.GREEN)
        for event in liquidity_events["detected_events"]:
            print_colored(f"  - {event['type']}: 强度={event.get('strength', 'N/A')}", Colors.GREEN)

    return combined_evaluation


def calculate_optimal_entry_with_liquidity(df, current_price, original_entry, liquidity_events):
    """
    根据流动性事件调整最佳入场价格

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格
        original_entry: 原始的最佳入场价格信息
        liquidity_events: 流动性事件分析结果

    返回:
        调整后的最佳入场价格信息
    """
    # 复制原始入场信息
    adjusted_entry = original_entry.copy()

    # 默认使用原始入场价格
    optimal_price = original_entry["price"]
    wait_needed = original_entry["wait_needed"]
    wait_description = original_entry["wait_description"]

    # 流动性事件入场价格候选
    liquidity_prices = []

    # 1. 流动性吸收事件
    if liquidity_events["liquidity_absorption"]["detected"]:
        la = liquidity_events["liquidity_absorption"]
        la_price = la["level"]
        la_direction = la["direction"]

        if la_direction == "high":
            # 高点流动性吸收，价格可能继续下跌，稍低入场
            la_entry = la_price * 0.995  # 略低于吸收水平0.5%
            liquidity_prices.append({
                "price": la_entry,
                "source": "liquidity_absorption_high",
                "quality": la["strength"],
                "description": f"高点流动性吸收后回踩价格 {la_entry:.6f}"
            })
        elif la_direction == "low":
            # 低点流动性吸收，价格可能继续上涨，稍高入场
            la_entry = la_price * 1.005  # 略高于吸收水平0.5%
            liquidity_prices.append({
                "price": la_entry,
                "source": "liquidity_absorption_low",
                "quality": la["strength"],
                "description": f"低点流动性吸收后回踩价格 {la_entry:.6f}"
            })

    # 2. 止损猎杀事件
    if liquidity_events["stop_hunt"]["detected"]:
        sh = liquidity_events["stop_hunt"]
        sh_level = sh["level"]
        sh_direction = sh["direction"]

        if sh_direction == "down_up":  # 向下突破后反转
            sh_entry = sh_level * 0.997  # 略低于猎杀水平
            liquidity_prices.append({
                "price": sh_entry,
                "source": "stop_hunt_down_up",
                "quality": sh["strength"],
                "description": f"下方止损猎杀后回测价格 {sh_entry:.6f}"
            })
        elif sh_direction == "up_down":  # 向上突破后反转
            sh_entry = sh_level * 1.003  # 略高于猎杀水平
            liquidity_prices.append({
                "price": sh_entry,
                "source": "stop_hunt_up_down",
                "quality": sh["strength"],
                "description": f"上方止损猎杀后回测价格 {sh_entry:.6f}"
            })

    # 3. 价格不平衡区域
    if liquidity_events["imbalance"]["detected"] and liquidity_events["imbalance"]["recent_zone"]:
        imb_zone = liquidity_events["imbalance"]["recent_zone"]

        if not imb_zone["filled"]:
            if imb_zone["type"] == "bullish":
                # 买方不平衡区域，入场点在区域上缘
                imb_entry = imb_zone["start_price"]
                liquidity_prices.append({
                    "price": imb_entry,
                    "source": "imbalance_bullish",
                    "quality": 7.0,
                    "description": f"买方不平衡区域入场点 {imb_entry:.6f}"
                })
            else:  # bearish
                # 卖方不平衡区域，入场点在区域下缘
                imb_entry = imb_zone["end_price"]
                liquidity_prices.append({
                    "price": imb_entry,
                    "source": "imbalance_bearish",
                    "quality": 7.0,
                    "description": f"卖方不平衡区域入场点 {imb_entry:.6f}"
                })

    # 4. 机构交易意图
    if liquidity_events["institutional_intent"]["detected"]:
        intent = liquidity_events["institutional_intent"]

        for ev in intent["evidence"]:
            if ev["type"] in ["multiple_tests_of_support", "multiple_tests_of_resistance"]:
                level = ev["level"]

                if ev["type"] == "multiple_tests_of_support":
                    # 多次测试的支撑位，入场点略高于支撑
                    int_entry = level * 1.003
                    liquidity_prices.append({
                        "price": int_entry,
                        "source": "institutional_support",
                        "quality": ev["test_count"] * 1.5,
                        "description": f"机构多次测试支撑位入场点 {int_entry:.6f}"
                    })
                else:  # resistance
                    # 多次测试的阻力位，入场点略低于阻力
                    int_entry = level * 0.997
                    liquidity_prices.append({
                        "price": int_entry,
                        "source": "institutional_resistance",
                        "quality": ev["test_count"] * 1.5,
                        "description": f"机构多次测试阻力位入场点 {int_entry:.6f}"
                    })

    # 选择最佳的流动性事件入场价格
    if liquidity_prices:
        # 按质量排序
        liquidity_prices.sort(key=lambda x: x["quality"], reverse=True)
        best_liquidity_price = liquidity_prices[0]

        # 与原始入场价格比较
        original_quality = original_entry.get("quality", 5.0)

        # 如果流动性事件入场质量更高，使用流动性事件入场
        if best_liquidity_price["quality"] > original_quality:
            optimal_price = best_liquidity_price["price"]
            wait_description = best_liquidity_price["description"]

            # 判断是否需要等待
            price_diff_pct = abs(optimal_price - current_price) / current_price * 100
            wait_needed = price_diff_pct > 0.3  # 如果差异超过0.3%，则需要等待

        # 添加流动性事件分析结果
        adjusted_entry["liquidity_price_candidates"] = liquidity_prices

    # 更新入场信息
    adjusted_entry["price"] = optimal_price
    adjusted_entry["wait_needed"] = wait_needed
    adjusted_entry["wait_description"] = wait_description

    return adjusted_entry