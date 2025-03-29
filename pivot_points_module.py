"""
支点(Pivot Points)分析模块
提供经典、Woodie和Camarilla三种支点计算方法，用于确定市场潜在支撑和阻力位
"""

import pandas as pd
import numpy as np


def calculate_pivot_points(df, method='classic'):
    """
    计算不同类型的支点

    参数:
    - df: 价格数据 DataFrame
    - method: 支点计算方法 ('classic', 'woodie', 'camarilla')

    返回:
    - DataFrame，添加支点相关指标
    """
    # 获取前一个周期的高、低、收盘价
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_close = df['close'].shift(1)

    if method == 'classic':
        # 经典支点计算方法
        pivot_point = (prev_high + prev_low + prev_close) / 3

        # 支撑位和阻力位
        r1 = (pivot_point * 2) - prev_low
        s1 = (pivot_point * 2) - prev_high

        r2 = pivot_point + (prev_high - prev_low)
        s2 = pivot_point - (prev_high - prev_low)

        r3 = prev_high + 2 * (pivot_point - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot_point)

    elif method == 'woodie':
        # Woodie's 支点方法
        pivot_point = (prev_high + prev_low + 2 * prev_close) / 4

        r1 = (2 * pivot_point) - prev_low
        s1 = (2 * pivot_point) - prev_high

        r2 = pivot_point + (prev_high - prev_low)
        s2 = pivot_point - (prev_high - prev_low)

        r3 = pivot_point + 2 * (prev_high - prev_low)
        s3 = pivot_point - 2 * (prev_high - prev_low)

    elif method == 'camarilla':
        # Camarilla 支点方法
        pivot_point = (prev_high + prev_low + prev_close) / 3

        r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
        s1 = prev_close - (prev_high - prev_low) * 1.1 / 12

        r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
        s2 = prev_close - (prev_high - prev_low) * 1.1 / 6

        r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    else:
        raise ValueError(f"不支持的支点计算方法: {method}")

    # 添加到 DataFrame
    df[f'{method.capitalize()}_PP'] = pivot_point
    df[f'{method.capitalize()}_R1'] = r1
    df[f'{method.capitalize()}_S1'] = s1
    df[f'{method.capitalize()}_R2'] = r2
    df[f'{method.capitalize()}_S2'] = s2
    df[f'{method.capitalize()}_R3'] = r3
    df[f'{method.capitalize()}_S3'] = s3

    # 价格相对支点的位置
    range_val = (prev_high - prev_low).replace(0, np.finfo(float).eps)  # 避免除以零
    df[f'{method.capitalize()}_Position'] = (df['close'] - pivot_point) / range_val

    return df


def analyze_pivot_point_strategy(df, method='classic'):
    """
    分析基于支点的交易策略

    参数:
    - df: 价格数据 DataFrame
    - method: 支点计算方法

    返回:
    - 交易信号和详细信息
    """
    # 首先确保支点已计算
    if f'{method.capitalize()}_PP' not in df.columns:
        df = calculate_pivot_points(df, method)

    # 获取最新的支点信息
    latest = df.iloc[-1]
    current_price = latest['close']

    # 支点信息
    pp = latest[f'{method.capitalize()}_PP']
    r1 = latest[f'{method.capitalize()}_R1']
    s1 = latest[f'{method.capitalize()}_S1']
    r2 = latest[f'{method.capitalize()}_R2']
    s2 = latest[f'{method.capitalize()}_S2']

    # 交易信号判断
    signal = "NEUTRAL"
    confidence = 0
    reason = ""

    # 价格位于支撑位下方 - 潜在买入信号
    if current_price < s1:
        signal = "BUY"
        # 计算置信度：价格越接近支撑位S2，置信度越高
        if current_price < s2:
            confidence = 0.8  # 非常接近S2
            reason = f"价格 ({current_price:.4f}) 低于第二支撑位 S2 ({s2:.4f})"
        else:
            confidence = 0.5  # 在S1和S2之间
            reason = f"价格 ({current_price:.4f}) 在第一支撑位 S1 ({s1:.4f}) 和第二支撑位 S2 ({s2:.4f}) 之间"

    # 价格位于阻力位上方 - 潜在卖出信号
    elif current_price > r1:
        signal = "SELL"
        # 计算置信度：价格越接近阻力位R2，置信度越高
        if current_price > r2:
            confidence = 0.8  # 非常接近R2
            reason = f"价格 ({current_price:.4f}) 高于第二阻力位 R2 ({r2:.4f})"
        else:
            confidence = 0.5  # 在R1和R2之间
            reason = f"价格 ({current_price:.4f}) 在第一阻力位 R1 ({r1:.4f}) 和第二阻力位 R2 ({r2:.4f}) 之间"

    # 价格在支点附近
    elif abs(current_price - pp) / pp < 0.005:  # 0.5%以内
        # 价格在支点附近，判断短期趋势
        if len(df) > 5:  # 确保有足够的数据计算短期趋势
            recent_trend = df['close'].iloc[-5:].diff().mean()
            if recent_trend > 0:
                signal = "BUY"
                confidence = 0.3
                reason = f"价格 ({current_price:.4f}) 接近支点 ({pp:.4f})，短期趋势向上"
            elif recent_trend < 0:
                signal = "SELL"
                confidence = 0.3
                reason = f"价格 ({current_price:.4f}) 接近支点 ({pp:.4f})，短期趋势向下"
            else:
                reason = f"价格 ({current_price:.4f}) 接近支点 ({pp:.4f})，趋势不明确"
        else:
            reason = f"价格 ({current_price:.4f}) 接近支点 ({pp:.4f})"

    # 价格在S1和R1之间
    else:
        # 判断位置偏向性
        if current_price > pp:
            signal = "BUY" if current_price > (pp + r1) / 2 else "NEUTRAL"
            confidence = 0.2
            reason = f"价格 ({current_price:.4f}) 在支点 ({pp:.4f}) 和第一阻力位 R1 ({r1:.4f}) 之间"
        else:
            signal = "SELL" if current_price < (pp + s1) / 2 else "NEUTRAL"
            confidence = 0.2
            reason = f"价格 ({current_price:.4f}) 在支点 ({pp:.4f}) 和第一支撑位 S1 ({s1:.4f}) 之间"

    return {
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "pivot_point": pp,
        "support_1": s1,
        "resistance_1": r1,
        "support_2": s2,
        "resistance_2": r2,
        "method": method
    }


def get_pivot_points_quality_score(df, method='classic'):
    """
    计算基于支点的质量评分

    参数:
    - df: 价格数据DataFrame
    - method: 支点计算方法

    返回:
    - 质量评分和评分分解
    """
    # 确保支点已计算
    if f'{method.capitalize()}_PP' not in df.columns:
        df = calculate_pivot_points(df, method)

    # 获取最新数据
    latest = df.iloc[-1]
    current_price = latest['close']

    # 支点信息
    pp = latest[f'{method.capitalize()}_PP']
    r1 = latest[f'{method.capitalize()}_R1']
    s1 = latest[f'{method.capitalize()}_S1']
    r2 = latest[f'{method.capitalize()}_R2']
    s2 = latest[f'{method.capitalize()}_S2']
    position = latest[f'{method.capitalize()}_Position']

    # 基础评分
    base_score = 5.0
    score_components = {
        "base_score": base_score
    }

    # 1. 价格位置评分 (0-3分)
    position_score = 0
    if current_price < s2:  # 低于S2，强支撑区
        position_score = 3.0
        position_note = "价格低于第二支撑位(S2)"
    elif current_price < s1:  # 在S1和S2之间
        position_score = 2.0
        position_note = "价格在第一和第二支撑位之间(S1-S2)"
    elif current_price > r2:  # 高于R2，强阻力区
        position_score = -3.0
        position_note = "价格高于第二阻力位(R2)"
    elif current_price > r1:  # 在R1和R2之间
        position_score = -2.0
        position_note = "价格在第一和第二阻力位之间(R1-R2)"
    elif abs(current_price - pp) / pp < 0.005:  # 接近支点
        position_score = 0
        position_note = "价格接近支点(PP)"
    elif current_price > pp:  # 在PP和R1之间
        position_score = -1.0
        position_note = "价格在支点和第一阻力位之间(PP-R1)"
    else:  # 在PP和S1之间
        position_score = 1.0
        position_note = "价格在支点和第一支撑位之间(PP-S1)"

    score_components["position_score"] = position_score
    score_components["position_note"] = position_note

    # 2. 历史支撑阻力效果评分 (0-2分)
    if len(df) > 20:
        # 检查过去N根K线中支撑阻力位的有效性
        validation_window = df.iloc[-20:].copy()

        # 检查价格是否在接近支撑位时反弹
        support_effectiveness = 0
        for i in range(1, len(validation_window) - 1):
            price = validation_window['close'].iloc[i]
            s1_val = validation_window[f'{method.capitalize()}_S1'].iloc[i]
            s2_val = validation_window[f'{method.capitalize()}_S2'].iloc[i]

            # 如果价格接近支撑位
            if price < s1_val and price > s2_val:
                # 检查后续价格是否上涨
                if validation_window['close'].iloc[i + 1] > price:
                    support_effectiveness += 1

        # 检查价格是否在接近阻力位时回落
        resistance_effectiveness = 0
        for i in range(1, len(validation_window) - 1):
            price = validation_window['close'].iloc[i]
            r1_val = validation_window[f'{method.capitalize()}_R1'].iloc[i]
            r2_val = validation_window[f'{method.capitalize()}_R2'].iloc[i]

            # 如果价格接近阻力位
            if price > r1_val and price < r2_val:
                # 检查后续价格是否下跌
                if validation_window['close'].iloc[i + 1] < price:
                    resistance_effectiveness += 1

        # 计算历史有效性评分
        effectiveness_score = min(2.0, (support_effectiveness + resistance_effectiveness) / 5)
        effectiveness_note = f"历史支撑/阻力有效次数: {support_effectiveness}/{resistance_effectiveness}"
    else:
        effectiveness_score = 1.0  # 数据不足时给予中等评分
        effectiveness_note = "历史数据不足，无法评估支撑/阻力的有效性"

    score_components["effectiveness_score"] = effectiveness_score
    score_components["effectiveness_note"] = effectiveness_note

    # 3. 支撑阻力区间宽度评分 (0-1分)
    # 区间越窄，评分越高
    range_ratio = abs((r1 - s1) / current_price)
    if range_ratio < 0.02:  # 范围小于2%
        range_score = 1.0
        range_note = "支撑/阻力范围非常紧密(<2%)"
    elif range_ratio < 0.05:  # 范围小于5%
        range_score = 0.7
        range_note = "支撑/阻力范围适中(2-5%)"
    else:  # 范围大于5%
        range_score = 0.3
        range_note = f"支撑/阻力范围较宽(>{range_ratio * 100:.1f}%)"

    score_components["range_score"] = range_score
    score_components["range_note"] = range_note

    # 总分计算
    total_score = base_score + position_score + effectiveness_score + range_score
    # 确保分数在0-10范围内
    total_score = max(0, min(10, total_score))

    # 返回结果
    return {
        "score": total_score,
        "components": score_components
    }