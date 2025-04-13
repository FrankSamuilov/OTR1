import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from logger_utils import Colors, print_colored
from indicators_module import (
    find_swing_points,
    calculate_fibonacci_retracements,
    get_smc_trend_and_duration
)


def evaluate_price_position_conditions(df, current_price=None):
    """
    综合评估价格位置条件，包括:
    1. 高质量订单块
    2. 关键斐波那契回撤水平
    3. 均线支撑/阻力位置
    4. SMC关键价格结构位置

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格(如果为None则使用最新收盘价)

    返回:
        dict: 综合评估结果
    """
    if current_price is None:
        current_price = df['close'].iloc[-1]

    print_colored(f"当前价格: {current_price:.6f}", Colors.INFO)

    # 1. 高质量订单块分析
    order_blocks = identify_high_quality_order_blocks(df)
    print_colored(f"发现 {len(order_blocks)} 个订单块", Colors.INFO)

    # 识别与当前价格相关的订单块
    relevant_blocks = []
    for ob in order_blocks:
        # 检查价格与订单块的关系
        price_in_block = ob['price_low'] <= current_price <= ob['price_high']
        price_near_block = (abs(current_price - ob['price_mid']) / current_price < 0.02)  # 2%以内

        if price_in_block or price_near_block:
            relevant_blocks.append(ob)
            print_colored(
                f"价格在订单块附近: {ob['type']}, 质量: {ob['quality_score']:.1f}/10, "
                f"价格范围: {ob['price_low']:.6f}-{ob['price_high']:.6f}",
                Colors.GREEN
            )

    # 2. 斐波那契水平分析
    fib_analysis = analyze_fibonacci_levels(df, current_price)

    if fib_analysis["valid"]:
        near_level = fib_analysis["near_levels"][0]
        print_colored(
            f"价格接近斐波那契水平 {near_level['level']} ({near_level['value']:.6f}), "
            f"质量: {fib_analysis['quality_score']:.1f}/10",
            Colors.GREEN
        )
    else:
        print_colored("价格未接近关键斐波那契水平", Colors.YELLOW)

    # 3. 均线分析
    ma_analysis = analyze_ma_support_resistance(df, current_price)

    if ma_analysis["near_mas"]:
        print_colored(
            f"价格接近均线: {', '.join([ma['name'] for ma in ma_analysis['near_mas']])}, "
            f"质量: {ma_analysis['quality_score']:.1f}/10",
            Colors.GREEN
        )
    else:
        above_mas = [ma["name"] for ma in ma_analysis["above_mas"]]
        below_mas = [ma["name"] for ma in ma_analysis["below_mas"]]
        print_colored(
            f"价格上方均线: {', '.join(above_mas) if above_mas else '无'}, "
            f"下方均线: {', '.join(below_mas) if below_mas else '无'}, "
            f"质量: {ma_analysis['quality_score']:.1f}/10",
            Colors.INFO
        )

    # 4. SMC价格结构分析
    smc_analysis = analyze_smc_price_structure(df, current_price)

    if smc_analysis["valid"]:
        print_colored(
            f"SMC结构: {smc_analysis['structure_type']}, "
            f"订单块测试: {'是' if smc_analysis['ob_tested'] else '否'}, "
            f"质量: {smc_analysis['quality_score']:.1f}/10",
            Colors.GREEN if smc_analysis["quality_score"] > 5 else Colors.YELLOW
        )
    else:
        print_colored("无法识别清晰的SMC价格结构", Colors.YELLOW)

    # 综合评分
    total_weight = 0
    weighted_sum = 0

    # 订单块权重
    ob_score = max([ob["quality_score"] for ob in relevant_blocks]) if relevant_blocks else 0
    ob_weight = 0.3  # 30%权重
    weighted_sum += ob_score * ob_weight
    total_weight += ob_weight

    # 斐波那契权重
    fib_weight = 0.25  # 25%权重
    weighted_sum += fib_analysis["quality_score"] * fib_weight
    total_weight += fib_weight

    # 均线权重
    ma_weight = 0.2  # 20%权重
    weighted_sum += ma_analysis["quality_score"] * ma_weight
    total_weight += ma_weight

    # SMC结构权重
    smc_weight = 0.25  # 25%权重
    weighted_sum += smc_analysis["quality_score"] * smc_weight
    total_weight += smc_weight

    # 最终综合评分
    final_score = weighted_sum / total_weight if total_weight > 0 else 0

    # 生成综合评估
    evaluation = {
        "score": final_score,
        "order_blocks": {
            "relevant": relevant_blocks,
            "score": ob_score,
            "all": order_blocks
        },
        "fibonacci": fib_analysis,
        "moving_averages": ma_analysis,
        "smc_structure": smc_analysis,
        "final_evaluation": _interpret_price_position_score(final_score),
        "best_entry_price": _calculate_optimal_entry_price(
            df, current_price, relevant_blocks, fib_analysis, ma_analysis, smc_analysis
        )
    }

    # 打印最终评估
    print_colored("\n==== 价格位置条件综合评估 ====", Colors.BLUE + Colors.BOLD)
    print_colored(f"综合评分: {final_score:.2f}/10",
                  Colors.GREEN if final_score >= 7 else
                  Colors.YELLOW if final_score >= 5 else
                  Colors.RED)
    print_colored(f"评估结果: {evaluation['final_evaluation']['description']}",
                  Colors.GREEN if final_score >= 7 else
                  Colors.YELLOW if final_score >= 5 else
                  Colors.RED)
    print_colored(f"最佳入场价格: {evaluation['best_entry_price']['price']:.6f}", Colors.INFO)
    print_colored(f"等待建议: {evaluation['best_entry_price']['wait_description']}", Colors.INFO)

    return evaluation


def _interpret_price_position_score(score):
    """解释价格位置评分"""
    if score >= 8.5:
        return {
            "quality": "excellent",
            "description": "极佳入场位置，多种因素强烈支持",
            "action": "immediate_entry"
        }
    elif score >= 7.0:
        return {
            "quality": "very_good",
            "description": "非常好的入场位置，大多数因素支持",
            "action": "strong_entry"
        }
    elif score >= 5.5:
        return {
            "quality": "good",
            "description": "良好入场位置，部分因素支持",
            "action": "consider_entry"
        }
    elif score >= 4.0:
        return {
            "quality": "fair",
            "description": "一般入场位置，建议等待更好机会",
            "action": "wait_better"
        }
    else:
        return {
            "quality": "poor",
            "description": "较差入场位置，不建议入场",
            "action": "avoid"
        }


def _calculate_optimal_entry_price(df, current_price, order_blocks, fib_analysis, ma_analysis, smc_analysis):
    """计算最佳入场价格"""
    # 默认使用当前价格
    optimal_price = current_price
    wait_needed = False
    wait_description = "可以立即入场"

    # 查找最佳入场价位
    candidate_prices = []

    # 1. 订单块中点
    if order_blocks:
        best_block = max(order_blocks, key=lambda x: x["quality_score"])
        candidate_prices.append({
            "price": best_block["price_mid"],
            "source": "order_block",
            "quality": best_block["quality_score"],
            "description": f"优质订单块中点 (质量: {best_block['quality_score']:.1f}/10)"
        })

    # 2. 最近的斐波那契水平
    if fib_analysis["valid"] and fib_analysis["near_levels"]:
        best_fib = fib_analysis["near_levels"][0]
        candidate_prices.append({
            "price": best_fib["value"],
            "source": "fibonacci",
            "quality": fib_analysis["quality_score"],
            "description": f"斐波那契 {best_fib['level']} 回撤位 (质量: {fib_analysis['quality_score']:.1f}/10)"
        })

    # 3. 最近的均线支撑位
    if ma_analysis["near_mas"]:
        best_ma = ma_analysis["near_mas"][0]
        candidate_prices.append({
            "price": best_ma["value"],
            "source": "moving_average",
            "quality": ma_analysis["quality_score"],
            "description": f"{best_ma['name']} 均线支撑 (质量: {ma_analysis['quality_score']:.1f}/10)"
        })

    # 4. SMC关键水平
    if smc_analysis["valid"] and smc_analysis["key_levels"]:
        # 找出最强的支撑位
        best_support = None
        for level in smc_analysis["key_levels"]:
            if level["type"] == "support":
                if not best_support or level["strength"] > best_support["strength"]:
                    best_support = level

        if best_support:
            candidate_prices.append({
                "price": best_support["price"],
                "source": "smc_support",
                "quality": smc_analysis["quality_score"] * (best_support["strength"] / 3),
                "description": f"SMC支撑位 (强度: {best_support['strength']}/3)"
            })

    # 选择最佳候选价格
    if candidate_prices:
        best_candidate = max(candidate_prices, key=lambda x: x["quality"])
        optimal_price = best_candidate["price"]

        # 判断是否需要等待
        price_diff_pct = abs(optimal_price - current_price) / current_price * 100

        if price_diff_pct > 0.5:  # 如果相差超过0.5%
            wait_needed = True
            wait_description = f"建议等待价格到达 {optimal_price:.6f} ({best_candidate['description']})"
        else:
            wait_description = f"当前价格接近最佳入场点: {best_candidate['description']}"

    return {
        "price": optimal_price,
        "wait_needed": wait_needed,
        "wait_description": wait_description,
        "candidates": candidate_prices
    }


def analyze_smc_price_structure(df, current_price):
    """
    分析SMC关键价格结构位置

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格

    返回:
        dict: 包含SMC价格结构分析结果
    """
    smc_results = {
        "valid": False,
        "structure_type": None,
        "quality_score": 0.0,
        "key_levels": [],
        "bos_points": [],  # Break of Structure points
        "ob_tested": False  # Order Block tested
    }

    # 获取摆动点
    swing_highs, swing_lows = find_swing_points(df)

    if not swing_highs or not swing_lows:
        return smc_results

    # 获取最近的摆动点
    recent_highs = sorted([h for h in swing_highs if h > current_price])
    recent_lows = sorted([l for l in swing_lows if l < current_price], reverse=True)

    # 找出结构变更点 (Break of Structure, BOS)
    bos_points = []
    for i in range(3, len(df)):
        # 向上突破
        if (df['high'].iloc[i] > df['high'].iloc[i - 1] > df['high'].iloc[i - 2] and
                df['low'].iloc[i] > df['low'].iloc[i - 1]):
            bos_points.append({
                "index": i,
                "price": df['high'].iloc[i],
                "type": "bullish"
            })

        # 向下突破
        if (df['low'].iloc[i] < df['low'].iloc[i - 1] < df['low'].iloc[i - 2] and
                df['high'].iloc[i] < df['high'].iloc[i - 1]):
            bos_points.append({
                "index": i,
                "price": df['low'].iloc[i],
                "type": "bearish"
            })

    # 确定市场结构类型
    current_structure = None
    if recent_highs and recent_lows:
        # 检查高点低点序列
        higher_highs = True
        higher_lows = True

        # 需要至少2个高点和低点才能确定结构
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # 检查更高的高点
            higher_highs = recent_highs[0] > recent_highs[1]
            # 检查更高的低点
            higher_lows = recent_lows[0] > recent_lows[1]

            if higher_highs and higher_lows:
                current_structure = "uptrend"
            elif not higher_highs and not higher_lows:
                current_structure = "downtrend"
            else:
                current_structure = "consolidation"

    # 填充关键水平
    key_levels = []

    # 添加最近的摆动高点作为阻力位
    for i, high in enumerate(recent_highs[:3]):  # 最多3个
        key_levels.append({
            "type": "resistance",
            "price": high,
            "strength": 3 - i  # 第一个最强
        })

    # 添加最近的摆动低点作为支撑位
    for i, low in enumerate(recent_lows[:3]):  # 最多3个
        key_levels.append({
            "type": "support",
            "price": low,
            "strength": 3 - i  # 第一个最强
        })

    # 检查订单块测试
    order_blocks = identify_high_quality_order_blocks(df)
    ob_tested = False

    for ob in order_blocks:
        # 如果价格在订单块范围内
        if ob['price_low'] <= current_price <= ob['price_high']:
            ob_tested = True
            # 添加到关键水平
            key_levels.append({
                "type": "order_block",
                "price_high": ob['price_high'],
                "price_low": ob['price_low'],
                "score": ob['quality_score']
            })

    # 评估质量
    quality_score = 0.0

    # 1. 明确市场结构加分
    if current_structure:
        quality_score += 3.0

    # 2. 价格在重要结构水平附近加分
    near_key_level = False
    for level in key_levels:
        if level["type"] in ["support", "resistance"]:
            if abs(current_price - level["price"]) / current_price < 0.01:  # 1%以内
                near_key_level = True
                quality_score += level["strength"]

    # 3. 测试订单块加分
    if ob_tested:
        quality_score += 3.0

    # 4. 有结构突破点加分
    if bos_points:
        quality_score += 2.0

    # 完成结果
    smc_results["valid"] = True
    smc_results["structure_type"] = current_structure
    smc_results["quality_score"] = min(10.0, quality_score)
    smc_results["key_levels"] = key_levels
    smc_results["bos_points"] = bos_points
    smc_results["ob_tested"] = ob_tested

    return smc_results


def analyze_ma_support_resistance(df, current_price):
    """
    分析当前价格与关键均线的支撑/阻力关系

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格

    返回:
        dict: 包含均线支撑阻力分析结果
    """
    ma_results = {
        "above_mas": [],
        "below_mas": [],
        "near_mas": [],
        "quality_score": 0.0,
        "ma_crossovers": []
    }

    # 检查常用均线
    ma_columns = [
        ("EMA5", 5), ("EMA20", 20), ("EMA50", 50), ("EMA200", 200),
        ("SMA10", 10), ("SMA50", 50), ("SMA200", 200)
    ]

    for ma_name, period in ma_columns:
        # 检查均线是否存在，不存在则尝试计算
        if ma_name not in df.columns:
            if ma_name.startswith("EMA"):
                df[ma_name] = df['close'].ewm(span=period).mean()
            elif ma_name.startswith("SMA"):
                df[ma_name] = df['close'].rolling(window=period).mean()

        # 获取均线值
        if ma_name in df.columns:
            ma_value = df[ma_name].iloc[-1]

            # 检查价格与均线的关系
            if abs(current_price - ma_value) / current_price < 0.005:  # 0.5%以内视为接近
                ma_results["near_mas"].append({
                    "name": ma_name,
                    "value": ma_value,
                    "distance": abs(current_price - ma_value) / current_price
                })
            elif current_price > ma_value:
                ma_results["above_mas"].append({
                    "name": ma_name,
                    "value": ma_value,
                    "distance": (current_price - ma_value) / current_price
                })
            else:
                ma_results["below_mas"].append({
                    "name": ma_name,
                    "value": ma_value,
                    "distance": (ma_value - current_price) / current_price
                })

    # 检查均线交叉
    if len(df) >= 3:
        # EMA交叉
        if "EMA5" in df.columns and "EMA20" in df.columns:
            current_cross = df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]
            prev_cross = df["EMA5"].iloc[-2] <= df["EMA20"].iloc[-2]

            if current_cross and prev_cross:
                ma_results["ma_crossovers"].append({
                    "type": "golden_cross",
                    "fast_ma": "EMA5",
                    "slow_ma": "EMA20"
                })
            elif not current_cross and not prev_cross:
                ma_results["ma_crossovers"].append({
                    "type": "death_cross",
                    "fast_ma": "EMA5",
                    "slow_ma": "EMA20"
                })

    # 评估质量
    quality_score = 0.0

    # 1. 接近均线加分
    if ma_results["near_mas"]:
        quality_score += 3.0
        # 接近重要均线额外加分
        for ma in ma_results["near_mas"]:
            if ma["name"] in ["EMA20", "EMA50", "SMA50", "EMA200", "SMA200"]:
                quality_score += 1.0

    # 2. 均线多头排列加分
    if len(ma_results["above_mas"]) > 2 and "EMA5" in [ma["name"] for ma in ma_results["above_mas"]]:
        quality_score += 2.0

    # 3. 多均线支撑加分
    if len(ma_results["below_mas"]) >= 3:
        quality_score += 2.0

    # 4. 均线交叉加分
    if any(cross["type"] == "golden_cross" for cross in ma_results["ma_crossovers"]):
        quality_score += 2.0

    ma_results["quality_score"] = min(10.0, quality_score)
    return ma_results


def analyze_fibonacci_levels(df, current_price, trend="auto"):
    """
    分析当前价格与斐波那契回撤水平的关系

    参数:
        df: 价格数据DataFrame
        current_price: 当前价格
        trend: 趋势方向 ("UP", "DOWN", "auto")

    返回:
        dict: 包含斐波那契分析结果
    """
    # 获取斐波那契水平
    fib_levels = calculate_fibonacci_retracements(df)

    if not fib_levels or len(fib_levels) < 3:
        return {"valid": False, "reason": "无法计算斐波那契水平"}

    # 如果趋势是自动检测，则基于SMC计算
    if trend == "auto":
        trend, _, _ = get_smc_trend_and_duration(df)

    # 获取关键斐波那契水平
    fib_236 = fib_levels[0]  # 0.236
    fib_382 = fib_levels[1]  # 0.382
    fib_500 = fib_levels[2]  # 0.500
    fib_618 = fib_levels[3] if len(fib_levels) > 3 else None  # 0.618
    fib_786 = fib_levels[4] if len(fib_levels) > 4 else None  # 0.786

    # 检查当前价格与斐波那契水平的关系
    near_levels = []
    threshold = 0.01  # 价格在1%范围内视为接近

    # 检查每个水平
    for level_name, level_value in [
        ("0.236", fib_236),
        ("0.382", fib_382),
        ("0.500", fib_500),
        ("0.618", fib_618),
        ("0.786", fib_786)
    ]:
        if level_value is not None and abs(current_price - level_value) / current_price < threshold:
            near_levels.append({
                "level": level_name,
                "value": level_value,
                "distance": abs(current_price - level_value) / current_price
            })

    # 黄金比例优先级排序
    if near_levels:
        near_levels.sort(key=lambda x: 0 if x["level"] in ["0.618", "0.382"] else
        1 if x["level"] in ["0.500"] else 2)

    # 评估质量
    quality_score = 0
    if near_levels:
        # 基础分数
        quality_score = 5.0

        # 黄金比率加分
        if near_levels[0]["level"] == "0.618":
            quality_score += 3.0
        elif near_levels[0]["level"] == "0.382":
            quality_score += 2.5
        elif near_levels[0]["level"] == "0.500":
            quality_score += 2.0
        else:
            quality_score += 1.0

        # 趋势方向加分
        if trend == "UP" and current_price > fib_500:
            quality_score += 1.0
        elif trend == "DOWN" and current_price < fib_500:
            quality_score += 1.0

    return {
        "valid": len(near_levels) > 0,
        "near_levels": near_levels,
        "quality_score": min(10.0, quality_score),
        "all_levels": fib_levels,
        "trend": trend
    }


def identify_high_quality_order_blocks(df, lookback=30, min_volume_ratio=1.3, max_price_deviation=0.5):
    """
    识别高质量订单块并评分

    参数:
        df: 价格数据DataFrame
        lookback: 回溯检查的K线数量
        min_volume_ratio: 最小成交量比率要求
        max_price_deviation: 最大价格波动要求(相对于ATR的比例)

    返回:
        订单块列表，每个包含位置、质量评分和特征
    """
    order_blocks = []

    # 确保数据足够
    if len(df) < lookback:
        return order_blocks

    # 获取ATR值用于相对波动判断
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (df['high'] - df['low']).mean()

    # 分析每个可能的订单块
    for i in range(len(df) - lookback, len(df) - 1):
        # 基本特征检查
        volume_ratio = df['volume'].iloc[i] / df['volume'].iloc[i - 5:i].mean() if i >= 5 else 1.0
        price_change = abs(df['close'].iloc[i] - df['close'].iloc[i - 1])
        price_deviation = price_change / atr

        # 检查是否符合订单块基本条件
        if volume_ratio >= min_volume_ratio and price_deviation <= max_price_deviation:
            # 确定订单块方向(多/空)
            is_bullish = df['close'].iloc[i] > df['open'].iloc[i]
            block_type = "bullish" if is_bullish else "bearish"

            # 计算质量评分 (1-10分)
            quality_score = _calculate_order_block_quality(df, i, volume_ratio, price_deviation)

            # 订单块价格范围
            ob_high = max(df['high'].iloc[i], df['high'].iloc[i - 1])
            ob_low = min(df['low'].iloc[i], df['low'].iloc[i - 1])

            # 添加到结果
            order_blocks.append({
                "index": i,
                "type": block_type,
                "quality_score": quality_score,
                "price_high": ob_high,
                "price_low": ob_low,
                "price_mid": (ob_high + ob_low) / 2,
                "volume_ratio": volume_ratio,
                "candle_time": df['time'].iloc[i] if 'time' in df.columns else None
            })

    # 按质量评分排序
    order_blocks.sort(key=lambda x: x['quality_score'], reverse=True)

    return order_blocks


def _calculate_order_block_quality(df, index, volume_ratio, price_deviation):
    """计算订单块质量评分"""
    score = 5.0  # 基础分

    # 1. 成交量评分 (0-3分)
    if volume_ratio > 2.5:
        volume_score = 3.0
    elif volume_ratio > 1.8:
        volume_score = 2.0
    elif volume_ratio > 1.3:
        volume_score = 1.0
    else:
        volume_score = 0.5

    # 2. 价格特征评分 (0-3分)
    # 优质订单块通常有较小的实体和较长的影线
    candle_body = abs(df['close'].iloc[index] - df['open'].iloc[index])
    candle_range = df['high'].iloc[index] - df['low'].iloc[index]

    if candle_range > 0:
        body_ratio = candle_body / candle_range
        if body_ratio < 0.3:  # 小实体长影线
            price_score = 3.0
        elif body_ratio < 0.5:
            price_score = 2.0
        elif body_ratio < 0.7:
            price_score = 1.0
        else:
            price_score = 0.5
    else:
        price_score = 0.0

    # 3. 位置评分 (0-4分)
    # 检查是否在关键支撑/阻力位附近
    position_score = 0.0

    # 检查是否靠近摆动高低点
    if 'Swing_Highs' in df.columns and 'Swing_Lows' in df.columns:
        high = df['high'].iloc[index]
        low = df['low'].iloc[index]

        # 获取最近的摆动点
        recent_swing_highs = df['Swing_Highs'].dropna().tail(3)
        recent_swing_lows = df['Swing_Lows'].dropna().tail(3)

        # 检查是否接近摆动点
        for swing_high in recent_swing_highs:
            if abs(high - swing_high) / high < 0.01:  # 1%以内
                position_score += 2.0
                break

        for swing_low in recent_swing_lows:
            if abs(low - swing_low) / low < 0.01:  # 1%以内
                position_score += 2.0
                break

    # 最终评分
    final_score = min(10.0, score + volume_score + price_score + position_score)
    return final_score