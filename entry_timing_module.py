"""
入场时机计算模块
计算最佳入场时机、入场条件和预期价格
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from logger_utils import Colors, print_colored


def calculate_entry_timing(df: pd.DataFrame, signal: str,
                           quality_score: float,
                           current_price: float) -> Dict[str, Any]:
    """
    计算最佳入场时机、条件和预期价格

    参数:
        df: 包含所有指标的DataFrame
        signal: 交易信号 ('BUY' 或 'SELL')
        quality_score: 质量评分
        current_price: 当前价格

    返回:
        包含入场时机详细信息的字典
    """
    print_colored("⏱️ 开始计算入场时机...", Colors.BLUE + Colors.BOLD)

    # 默认结果
    result = {
        "should_wait": True,
        "entry_type": "LIMIT",  # 默认使用限价单
        "entry_conditions": [],
        "expected_entry_price": current_price,
        "max_wait_time": 60,  # 默认最多等待60分钟
        "confidence": 0.5,
        "immediate_entry": False
    }

    try:
        # 获取最近价格数据
        recent_prices = df['close'].tail(10).values
        last_price = recent_prices[-1]

        # 计算当前波动性
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            volatility = atr / current_price * 100  # 以百分比表示
        else:
            std = np.std(recent_prices) / np.mean(recent_prices) * 100
            volatility = std  # 使用标准差作为波动性指标

        volatility_desc = "高" if volatility > 2 else "中" if volatility > 1 else "低"
        print_colored(f"当前波动性: {volatility:.2f}% ({volatility_desc})", Colors.INFO)

        # 基于信号和质量评分确定入场策略
        if signal == "BUY":
            # 买入策略
            entry_conditions = []

            # 1. 考虑支撑位
            support_levels = []

            # 从摆动低点寻找支撑位
            if 'Swing_Lows' in df.columns:
                recent_lows = df['Swing_Lows'].dropna().tail(3).values
                support_levels.extend([low for low in recent_lows if low < current_price])

            # 从支点寻找支撑位
            classic_s1 = df['Classic_S1'].iloc[-1] if 'Classic_S1' in df.columns else None
            if classic_s1 and classic_s1 < current_price:
                support_levels.append(classic_s1)

            # 从布林带寻找支撑位
            bb_lower = df['BB_Lower'].iloc[-1] if 'BB_Lower' in df.columns else None
            if bb_lower and bb_lower < current_price:
                support_levels.append(bb_lower)

            # 2. 确定入场条件
            if support_levels:
                # 找到最近的支撑位
                closest_support = max(support_levels)
                support_distance = (current_price - closest_support) / current_price * 100

                if support_distance < 0.5:  # 非常接近支撑位
                    entry_conditions.append(f"价格接近支撑位 {closest_support:.6f}，可以立即入场")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["expected_entry_price"] = current_price
                elif support_distance < 1.5:  # 接近但不是非常近
                    target_price = closest_support * 1.002  # 稍微高于支撑位
                    entry_conditions.append(f"等待价格回调至 {target_price:.6f} 附近（接近支撑位）")
                    result["expected_entry_price"] = target_price
                    result["max_wait_time"] = 180  # 等待时间延长
                else:
                    target_price = current_price * 0.995  # 轻微回调
                    entry_conditions.append(f"等待价格轻微回调至 {target_price:.6f}（当前价格的99.5%）")
                    result["expected_entry_price"] = target_price
            else:
                # 没有明确支撑位时的策略
                if quality_score >= 8.0:  # 质量评分很高
                    entry_conditions.append("质量评分高，可以市价入场")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["expected_entry_price"] = current_price
                else:
                    target_price = current_price * 0.997  # 轻微回调
                    entry_conditions.append(f"等待轻微回调至 {target_price:.6f}（当前价格的99.7%）")
                    result["expected_entry_price"] = target_price

            # 3. 考虑突破情况
            if 'BB_Upper' in df.columns:
                bb_upper = df['BB_Upper'].iloc[-1]
                if current_price > bb_upper:
                    # 价格突破布林带上轨
                    if (current_price - bb_upper) / bb_upper > 0.005:  # 显著突破
                        entry_conditions.append(f"价格已突破布林带上轨 {bb_upper:.6f}，等待回踩确认")
                        target_price = bb_upper * 1.001  # 略高于上轨
                        result["expected_entry_price"] = target_price
                        result["max_wait_time"] = 120  # 等待时间适中

            # 4. 考虑指标交叉信号
            stoch_cross_up = df['Stochastic_Cross_Up'].iloc[-1] if 'Stochastic_Cross_Up' in df.columns else 0
            if stoch_cross_up == 1:
                entry_conditions.append("随机指标形成金叉，可以考虑入场")
                result["confidence"] += 0.1
                if not result["immediate_entry"]:
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["expected_entry_price"] = current_price

            # 检查SAR反转信号
            if 'SAR_Trend_Change' in df.columns and df['SAR_Trend'].iloc[-1] == 1:
                if df['SAR_Trend_Change'].iloc[-1] > 0:
                    entry_conditions.append("SAR刚刚转为上升趋势，信号较强")
                    result["confidence"] += 0.15
                    if not result["immediate_entry"]:
                        result["immediate_entry"] = True
                        result["should_wait"] = False
                        result["entry_type"] = "MARKET"
                        result["expected_entry_price"] = current_price

            # 5. 基于波动性调整策略
            if volatility > 2.0:  # 高波动环境
                if result["entry_type"] == "LIMIT":
                    target_price = result["expected_entry_price"] * 0.98  # 更大的价格优惠
                    entry_conditions.append(f"高波动环境，可设置更低的限价单 {target_price:.6f}")
                    result["expected_entry_price"] = target_price
                if not entry_conditions:
                    entry_conditions.append("高波动环境，建议使用分批入场")

        else:  # SELL信号
            # 卖出策略
            entry_conditions = []

            # 1. 考虑阻力位
            resistance_levels = []

            # 从摆动高点寻找阻力位
            if 'Swing_Highs' in df.columns:
                recent_highs = df['Swing_Highs'].dropna().tail(3).values
                resistance_levels.extend([high for high in recent_highs if high > current_price])

            # 从支点寻找阻力位
            classic_r1 = df['Classic_R1'].iloc[-1] if 'Classic_R1' in df.columns else None
            if classic_r1 and classic_r1 > current_price:
                resistance_levels.append(classic_r1)

            # 从布林带寻找阻力位
            bb_upper = df['BB_Upper'].iloc[-1] if 'BB_Upper' in df.columns else None
            if bb_upper and bb_upper > current_price:
                resistance_levels.append(bb_upper)

            # 2. 确定入场条件
            if resistance_levels:
                # 找到最近的阻力位
                closest_resistance = min(resistance_levels)
                resistance_distance = (closest_resistance - current_price) / current_price * 100

                if resistance_distance < 0.5:  # 非常接近阻力位
                    entry_conditions.append(f"价格接近阻力位 {closest_resistance:.6f}，可以立即入场")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["expected_entry_price"] = current_price
                elif resistance_distance < 1.5:  # 接近但不是非常近
                    target_price = closest_resistance * 0.998  # 稍微低于阻力位
                    entry_conditions.append(f"等待价格反弹至 {target_price:.6f} 附近（接近阻力位）")
                    result["expected_entry_price"] = target_price
                    result["max_wait_time"] = 180  # 等待时间延长
                else:
                    target_price = current_price * 1.005  # 轻微反弹
                    entry_conditions.append(f"等待价格轻微反弹至 {target_price:.6f}（当前价格的100.5%）")
                    result["expected_entry_price"] = target_price
            else:
                # 没有明确阻力位时的策略
                if quality_score >= 8.0:  # 质量评分很高
                    entry_conditions.append("质量评分高，可以市价入场")
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["expected_entry_price"] = current_price
                else:
                    target_price = current_price * 1.003  # 轻微反弹
                    entry_conditions.append(f"等待轻微反弹至 {target_price:.6f}（当前价格的100.3%）")
                    result["expected_entry_price"] = target_price

            # 3. 考虑突破情况
            if 'BB_Lower' in df.columns:
                bb_lower = df['BB_Lower'].iloc[-1]
                if current_price < bb_lower:
                    # 价格突破布林带下轨
                    if (bb_lower - current_price) / bb_lower > 0.005:  # 显著突破
                        entry_conditions.append(f"价格已突破布林带下轨 {bb_lower:.6f}，等待回踩确认")
                        target_price = bb_lower * 0.999  # 略低于下轨
                        result["expected_entry_price"] = target_price
                        result["max_wait_time"] = 120  # 等待时间适中

            # 4. 考虑指标交叉信号
            stoch_cross_down = df['Stochastic_Cross_Down'].iloc[-1] if 'Stochastic_Cross_Down' in df.columns else 0
            if stoch_cross_down == 1:
                entry_conditions.append("随机指标形成死叉，可以考虑入场")
                result["confidence"] += 0.1
                if not result["immediate_entry"]:
                    result["immediate_entry"] = True
                    result["should_wait"] = False
                    result["entry_type"] = "MARKET"
                    result["expected_entry_price"] = current_price

            # 检查SAR反转信号
            if 'SAR_Trend_Change' in df.columns and df['SAR_Trend'].iloc[-1] == -1:
                if df['SAR_Trend_Change'].iloc[-1] > 0:
                    entry_conditions.append("SAR刚刚转为下降趋势，信号较强")
                    result["confidence"] += 0.15
                    if not result["immediate_entry"]:
                        result["immediate_entry"] = True
                        result["should_wait"] = False
                        result["entry_type"] = "MARKET"
                        result["expected_entry_price"] = current_price

            # 5. 基于波动性调整策略
            if volatility > 2.0:  # 高波动环境
                if result["entry_type"] == "LIMIT":
                    target_price = result["expected_entry_price"] * 1.02  # 更大的价格优惠
                    entry_conditions.append(f"高波动环境，可设置更高的限价单 {target_price:.6f}")
                    result["expected_entry_price"] = target_price
                if not entry_conditions:
                    entry_conditions.append("高波动环境，建议使用分批入场")

        # 6. 根据质量评分调整入场策略
        if quality_score >= 9.0 and not result["immediate_entry"]:
            entry_conditions.append("质量评分极高，建议立即市价入场")
            result["immediate_entry"] = True
            result["should_wait"] = False
            result["entry_type"] = "MARKET"
            result["expected_entry_price"] = current_price
            result["confidence"] += 0.2
        elif quality_score <= 5.0 and signal == "BUY":
            entry_conditions.append("质量评分较低，建议等待更好入场点或降低仓位")
            result["confidence"] -= 0.1
            result["max_wait_time"] = 30  # 缩短等待时间
        elif quality_score <= 5.0 and signal == "SELL":
            entry_conditions.append("质量评分较低，建议等待更好入场点或降低仓位")
            result["confidence"] -= 0.1
            result["max_wait_time"] = 30  # 缩短等待时间

        # 计算预期入场时间
        current_time = datetime.now()
        if result["should_wait"]:
            # 根据波动性估计到达目标价格的时间
            price_diff_pct = abs(result["expected_entry_price"] - current_price) / current_price * 100
            expected_minutes = min(result["max_wait_time"], max(15, int(price_diff_pct / volatility * 60)))
            expected_entry_time = current_time + timedelta(minutes=expected_minutes)
            result["expected_entry_minutes"] = expected_minutes
            result["expected_entry_time"] = expected_entry_time.strftime("%H:%M:%S")
        else:
            result["expected_entry_minutes"] = 0
            result["expected_entry_time"] = current_time.strftime("%H:%M:%S") + " (立即)"

        # 保存入场条件
        result["entry_conditions"] = entry_conditions

        # 调整入场条件文本
        if not entry_conditions:
            if result["immediate_entry"]:
                entry_conditions.append("综合分析建议立即市价入场")
            else:
                entry_conditions.append(f"无明确入场条件，建议等待价格达到 {result['expected_entry_price']:.6f}")

        # 打印结果
        condition_color = Colors.GREEN if result["immediate_entry"] else Colors.YELLOW
        print_colored("入场时机分析结果:", Colors.BLUE)
        for i, condition in enumerate(entry_conditions, 1):
            print_colored(f"{i}. {condition}", condition_color)

        wait_msg = "立即入场" if result["immediate_entry"] else f"等待 {result['expected_entry_minutes']} 分钟"
        print_colored(f"建议入场时间: {result['expected_entry_time']} ({wait_msg})", Colors.INFO)
        print_colored(f"预期入场价格: {result['expected_entry_price']:.6f}", Colors.INFO)
        print_colored(f"入场类型: {result['entry_type']}", Colors.INFO)
        print_colored(f"入场置信度: {result['confidence']:.2f}", Colors.INFO)

        return result
    except Exception as e:
        print_colored(f"❌ 计算入场时机失败: {e}", Colors.ERROR)
        result["error"] = str(e)
        result["entry_conditions"] = ["计算出错，建议采用默认市价入场策略"]
        result["expected_entry_time"] = datetime.now().strftime("%H:%M:%S") + " (立即)"
        return result


def detect_breakout_conditions(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """
    检测价格突破条件

    参数:
        df: 价格数据DataFrame
        lookback: 回溯检查的K线数量

    返回:
        突破信息字典
    """
    print_colored("🔍 检测价格突破条件...", Colors.BLUE)

    try:
        # 确保数据足够
        if len(df) < lookback + 5:
            return {
                "has_breakout": False,
                "direction": "NONE",
                "strength": 0,
                "description": "数据不足，无法检测突破"
            }

        result = {
            "has_breakout": False,
            "direction": "NONE",
            "strength": 0,
            "description": "",
            "breakout_details": []
        }

        # 获取最新价格和成交量
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0

        # 计算近期价格区间
        lookback_df = df.iloc[-lookback:-1]
        recent_high = lookback_df['high'].max()
        recent_low = lookback_df['low'].min()

        # 计算平均成交量
        avg_volume = lookback_df['volume'].mean() if 'volume' in df.columns else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # 检查技术指标
        has_bb = all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle'])
        has_pivot = 'Classic_PP' in df.columns

        breakout_details = []

        # 1. 检查价格区间突破
        upside_breakout = current_price > recent_high
        downside_breakout = current_price < recent_low

        if upside_breakout:
            strength = (current_price - recent_high) / recent_high * 100
            breakout_details.append({
                "type": "price_range",
                "direction": "UP",
                "description": f"价格突破近期高点 {recent_high:.6f}",
                "strength": strength
            })
        elif downside_breakout:
            strength = (recent_low - current_price) / recent_low * 100
            breakout_details.append({
                "type": "price_range",
                "direction": "DOWN",
                "description": f"价格跌破近期低点 {recent_low:.6f}",
                "strength": strength
            })

        # 2. 检查布林带突破
        if has_bb:
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            bb_width = (bb_upper - bb_lower) / df['BB_Middle'].iloc[-1]

            # 上轨突破
            if current_price > bb_upper:
                bb_breakout_strength = (current_price - bb_upper) / bb_upper * 100
                bb_width_factor = max(1, bb_width * 10)  # 窄的布林带突破更有意义
                bb_strength = bb_breakout_strength * bb_width_factor

                breakout_details.append({
                    "type": "bollinger_band",
                    "direction": "UP",
                    "description": f"价格突破布林带上轨 {bb_upper:.6f}",
                    "strength": bb_strength
                })

            # 下轨突破
            elif current_price < bb_lower:
                bb_breakout_strength = (bb_lower - current_price) / bb_lower * 100
                bb_width_factor = max(1, bb_width * 10)
                bb_strength = bb_breakout_strength * bb_width_factor

                breakout_details.append({
                    "type": "bollinger_band",
                    "direction": "DOWN",
                    "description": f"价格跌破布林带下轨 {bb_lower:.6f}",
                    "strength": bb_strength
                })

        # 3. 检查支点突破
        if has_pivot:
            pivot = df['Classic_PP'].iloc[-1]
            r1 = df['Classic_R1'].iloc[-1]
            s1 = df['Classic_S1'].iloc[-1]

            # 阻力突破
            if df['close'].iloc[-2] <= r1 and current_price > r1:
                pivot_strength = (current_price - r1) / r1 * 100
                breakout_details.append({
                    "type": "pivot_point",
                    "direction": "UP",
                    "description": f"价格突破R1阻力位 {r1:.6f}",
                    "strength": pivot_strength
                })

            # 支撑跌破
            elif df['close'].iloc[-2] >= s1 and current_price < s1:
                pivot_strength = (s1 - current_price) / s1 * 100
                breakout_details.append({
                    "type": "pivot_point",
                    "direction": "DOWN",
                    "description": f"价格跌破S1支撑位 {s1:.6f}",
                    "strength": pivot_strength
                })

        # 4. 检查动量指标
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            prev_rsi = df['RSI'].iloc[-2]

            if prev_rsi < 30 and rsi > 30:
                breakout_details.append({
                    "type": "indicator",
                    "direction": "UP",
                    "description": f"RSI从超卖区反弹 ({prev_rsi:.1f} -> {rsi:.1f})",
                    "strength": (rsi - prev_rsi) / 2
                })
            elif prev_rsi > 70 and rsi < 70:
                breakout_details.append({
                    "type": "indicator",
                    "direction": "DOWN",
                    "description": f"RSI从超买区回落 ({prev_rsi:.1f} -> {rsi:.1f})",
                    "strength": (prev_rsi - rsi) / 2
                })

        # 汇总结果
        if breakout_details:
            # 过滤出强度最高的突破
            strongest_breakout = max(breakout_details, key=lambda x: x.get("strength", 0))
            result["has_breakout"] = True
            result["direction"] = strongest_breakout["direction"]
            result["strength"] = strongest_breakout["strength"]
            result["description"] = strongest_breakout["description"]
            result["breakout_details"] = breakout_details

            # 考虑成交量
            if volume_ratio > 1.5:
                result["strength"] *= 1.2
                result["description"] += f"，成交量放大({volume_ratio:.1f}倍)"

            print_colored(f"检测到{result['direction']}方向突破:",
                          Colors.GREEN if result['direction'] == 'UP' else Colors.RED)
            print_colored(f"描述: {result['description']}", Colors.INFO)
            print_colored(f"强度: {result['strength']:.2f}", Colors.INFO)

            for detail in breakout_details:
                detail_dir = detail["direction"]
                detail_color = Colors.GREEN if detail_dir == "UP" else Colors.RED
                print_colored(
                    f"- {detail['type']}: {detail_color}{detail['description']}{Colors.RESET}, 强度: {detail['strength']:.2f}",
                    Colors.INFO)
        else:
            print_colored("未检测到明显突破", Colors.YELLOW)

        return result
    except Exception as e:
        print_colored(f"❌ 检测突破条件失败: {e}", Colors.ERROR)
        return {
            "has_breakout": False,
            "direction": "NONE",
            "strength": 0,
            "description": f"检测出错: {str(e)}",
            "error": str(e)
        }


def estimate_entry_execution_price(current_price: float, signal: str,
                                   order_type: str, market_impact: float = 0.001) -> float:
    """
    估计实际入场执行价格，考虑市场冲击和滑点

    参数:
        current_price: 当前价格
        signal: 交易信号 ('BUY' 或 'SELL')
        order_type: 订单类型 ('MARKET' 或 'LIMIT')
        market_impact: 市场冲击系数

    返回:
        估计的执行价格
    """
    if order_type == "LIMIT":
        # 限价单通常以指定价格成交
        return current_price

    # 市价单会有滑点
    if signal == "BUY":
        # 买入时价格通常会略高于当前价
        execution_price = current_price * (1 + market_impact)
    else:  # SELL
        # 卖出时价格通常会略低于当前价
        execution_price = current_price * (1 - market_impact)

    return execution_price