"""
指标计算模块 - 修复版本
包含威廉指标计算，趋势判断和各种技术指标实现
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from data_module import get_historical_data
import logging
from logger_setup import get_logger
# 修改导入以使用正确的模块名称
from logger_utils import (
    Colors, format_log, print_colored,
    log_indicator, log_trend, log_market_conditions
)

# 设置指标日志
logging.basicConfig(
    filename='logs/indicators.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
indicators_logger = logging.getLogger('indicators')


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算威廉指标 (Williams %R)

    参数:
        df: 包含OHLC数据的DataFrame
        period: 计算周期，默认14

    返回:
        df: 添加了威廉指标的DataFrame
    """
    try:
        if len(df) < period:
            print_colored(f"⚠️ 数据长度 {len(df)} 小于威廉指标周期 {period}", Colors.WARNING)
            return df

        # 计算最高价和最低价
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        # 计算威廉指标 %R = -100 * (H - C) / (H - L)
        df['Williams_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)

        # 在最后一行显示威廉指标值
        last_value = df['Williams_R'].iloc[-1]

        # 计算威廉指标的短期趋势
        williams_slope = 0
        if len(df) >= 5:
            recent_williams = df['Williams_R'].tail(5).values
            williams_slope = np.polyfit(range(len(recent_williams)), recent_williams, 1)[0]

            # 添加威廉指标的变化率
            df['Williams_R_Change'] = df['Williams_R'].diff()

            # 添加威廉指标的变化加速度
            df['Williams_R_Acceleration'] = df['Williams_R_Change'].diff()

        # 判断超买超卖状态
        if last_value <= -80:
            williams_state = "超卖"
            color = Colors.OVERSOLD
        elif last_value >= -20:
            williams_state = "超买"
            color = Colors.OVERBOUGHT
        else:
            williams_state = "中性"
            color = Colors.RESET

        # 判断威廉指标的趋势方向
        if williams_slope > 1.5:
            williams_trend = "强势上升"
            trend_indicator = "⬆️⬆️"
        elif williams_slope > 0.5:
            williams_trend = "上升"
            trend_indicator = "⬆️"
        elif williams_slope < -1.5:
            williams_trend = "强势下降"
            trend_indicator = "⬇️⬇️"
        elif williams_slope < -0.5:
            williams_trend = "下降"
            trend_indicator = "⬇️"
        else:
            williams_trend = "平稳"
            trend_indicator = "➡️"

        print_colored(f"📊 威廉指标(Williams %R): {color}{last_value:.2f}{Colors.RESET} ({williams_state})", color)
        print_colored(f"{trend_indicator} 威廉指标趋势: {williams_trend}, 斜率: {williams_slope:.4f}",
                      Colors.BLUE if williams_slope > 0 else Colors.RED if williams_slope < 0 else Colors.RESET)

        return df
    except Exception as e:
        print_colored(f"❌ 计算威廉指标失败: {e}", Colors.ERROR)
        indicators_logger.error(f"计算威廉指标失败: {e}")
        return df


def calculate_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    增强版超级趋势指标计算，支持不同参数的超级趋势并增加信号稳定性检查

    参数:
        df: 包含OHLC数据的DataFrame
        atr_period: ATR计算周期
        multiplier: ATR乘数

    返回:
        df: 添加了超级趋势指标的DataFrame
    """
    # 检查是否是递归调用
    is_recursive = 'Supertrend' in df.columns

    if not is_recursive:
        print_colored(f"计算超级趋势指标 - ATR周期: {atr_period}, 乘数: {multiplier}", Colors.INFO)

    try:
        high = df['high']
        low = df['low']
        close = df['close']

        # 确保已经计算了ATR
        if 'ATR' not in df.columns:
            # 计算真实范围（TR）
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            # 计算ATR
            atr = tr.rolling(atr_period).mean()
            df['ATR'] = atr
            if not is_recursive:
                print_colored(f"计算ATR完成，均值: {atr.mean():.6f}", Colors.INFO)
        else:
            atr = df['ATR']

        # 计算基本上轨和下轨
        upperband = ((high + low) / 2) + (multiplier * atr)
        lowerband = ((high + low) / 2) - (multiplier * atr)

        # 初始化超级趋势
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1表示看多，-1表示看空

        # 第一个值使用默认值
        supertrend.iloc[0] = lowerband.iloc[0]

        # 计算超级趋势和方向
        for i in range(1, len(df)):
            if close.iloc[i] > upperband.iloc[i - 1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < lowerband.iloc[i - 1]:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
            else:
                if direction.iloc[i - 1] == 1:
                    supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i - 1])
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i - 1])
                    direction.iloc[i] = -1

        # 添加稳定性检查 - 是否有足够的连续方向
        min_stable_periods = 3  # 至少需要连续3个周期保持同一方向

        # 初始化稳定性列
        if 'Supertrend_Stability' not in df.columns:
            df['Supertrend_Stability'] = pd.Series(1.0, index=df.index)

        # 遍历计算稳定性
        for i in range(min_stable_periods, len(df)):
            is_stable = True
            for j in range(1, min_stable_periods):
                if direction.iloc[i - j] != direction.iloc[i]:
                    is_stable = False
                    break

            # 使用loc而不是链式赋值，以避免警告
            df.loc[i, 'Supertrend_Stability'] = 1.0 if is_stable else 0.5

        # 计算信号变化点
        if not is_recursive:
            signal_changes = []
            last_direction = direction.iloc[0]
            for i in range(1, len(direction)):
                if direction.iloc[i] != last_direction:
                    signal_changes.append((i, "BUY" if direction.iloc[i] > 0 else "SELL"))
                    last_direction = direction.iloc[i]
                    print_colored(
                        f"超级趋势信号变化 - 索引: {i}, 方向: {'看多' if direction.iloc[i] > 0 else '看空'}",
                        Colors.GREEN if direction.iloc[i] > 0 else Colors.RED
                    )

        # 添加到DataFrame
        col_prefix = "" if is_recursive else ""
        df[f'{col_prefix}Supertrend'] = supertrend
        df[f'{col_prefix}Supertrend_Direction'] = direction

        # 增加快速超级趋势，使用较小的参数
        if multiplier == 3 and not is_recursive:
            # 计算快速超级趋势
            df_copy = df.copy()
            df_copy = calculate_supertrend(df_copy, atr_period=5, multiplier=2)
            df['Fast_Supertrend'] = df_copy['Supertrend']
            df['Fast_Supertrend_Direction'] = df_copy['Supertrend_Direction']

            # 计算慢速超级趋势
            df_copy = df.copy()
            df_copy = calculate_supertrend(df_copy, atr_period=15, multiplier=4)
            df['Slow_Supertrend'] = df_copy['Supertrend']
            df['Slow_Supertrend_Direction'] = df_copy['Supertrend_Direction']

            # 计算三重超级趋势一致性
            if 'Fast_Supertrend_Direction' in df.columns and 'Slow_Supertrend_Direction' in df.columns:
                df['Supertrend_Consensus'] = ((df['Supertrend_Direction'] == df['Fast_Supertrend_Direction']) &
                                              (df['Supertrend_Direction'] == df['Slow_Supertrend_Direction'])).astype(
                    float)

                # 计算共识百分比
                consensus_pct = df['Supertrend_Consensus'].mean() * 100
                consensus_count = df['Supertrend_Consensus'].sum()
                consensus_status = "高" if consensus_pct >= 80 else "中" if consensus_pct >= 50 else "低"

                print_colored(
                    f"超级趋势共识度: {consensus_pct:.1f}% ({consensus_status}) - "
                    f"一致 {int(consensus_count)}次, 不一致 {len(df) - int(consensus_count)}次",
                    Colors.GREEN if consensus_pct >= 80 else
                    Colors.YELLOW if consensus_pct >= 50 else
                    Colors.RED
                )

        # 计算信号强度 - 价格与超级趋势的距离
        df[f'{col_prefix}Supertrend_Strength'] = abs(df['close'] - supertrend) / df['ATR']

        if not is_recursive:
            last_dir = df['Supertrend_Direction'].iloc[-1]
            last_str = df['Supertrend_Strength'].iloc[-1]
            dir_text = "看多" if last_dir > 0 else "看空"
            dir_color = Colors.GREEN if last_dir > 0 else Colors.RED

            print_colored(
                f"超级趋势: {dir_color}{dir_text}{Colors.RESET}, "
                f"强度: {last_str:.2f}, 均值: {df['Supertrend_Strength'].mean():.2f}",
                Colors.INFO
            )

        return df
    except Exception as e:
        print_colored(f"❌ 计算超级趋势指标失败: {e}", Colors.ERROR)
        indicators_logger.error(f"计算超级趋势指标失败: {e}")
        return df


def calculate_smma(df: pd.DataFrame, period: int = 60) -> pd.DataFrame:
    """
    计算平滑移动平均线 (SMMA)

    参数:
        df: 包含收盘价的DataFrame
        period: 计算周期

    返回:
        df: 添加了SMMA的DataFrame
    """
    try:
        if len(df) < period:
            indicators_logger.warning(f"数据长度 {len(df)} 小于SMMA周期 {period}")
            print_colored(f"⚠️ 数据长度 {len(df)} 小于SMMA周期 {period}", Colors.WARNING)
            return df

        # 初始化SMMA为前N个周期的SMA
        smma = pd.Series(index=df.index)
        smma.iloc[:period] = df['close'].iloc[:period].mean()

        # 计算后续值: SMMA(t) = (SMMA(t-1) * (period-1) + close(t)) / period
        for i in range(period, len(df)):
            smma.iloc[i] = (smma.iloc[i - 1] * (period - 1) + df['close'].iloc[i]) / period

        # 添加到DataFrame
        col_name = f'SMMA{period}'
        df[col_name] = smma
        print_colored(f"计算SMMA{period}完成，最新值: {smma.iloc[-1]:.4f}", Colors.INFO)

        return df
    except Exception as e:
        indicators_logger.error(f"计算SMMA失败: {e}")
        print_colored(f"❌ 计算SMMA失败: {e}", Colors.ERROR)
        return df


def get_smc_trend_and_duration(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None,
                              logger: Optional[logging.Logger] = None) -> Tuple[str, int, Dict[str, Any]]:
    """
    计算SMC趋势和趋势持续时间（分钟），集成订单块和流动性
    优化版本：减少所需数据量，整合多种指标进行趋势判断

    参数:
        df: 包含OHLC数据和指标的DataFrame
        config: 配置参数
        logger: 日志对象

    返回:
        trend: 趋势方向 ("UP", "DOWN", "NEUTRAL")
        duration: 趋势持续时间（分钟）
        trend_info: 趋势详细信息字典
    """
    if logger is None:
        logger = get_logger()
    if config is None:
        config = {"TREND_DURATION_THRESHOLD": 1440}

    if len(df) < 8 or 'high' not in df.columns or 'low' not in df.columns:
        print_colored("⚠️ 数据不足，无法分析趋势", Colors.WARNING)
        return "NEUTRAL", 0, {"confidence": "无", "reason": "数据不足"}

    # 准备趋势信息字典
    trend_info = {
        "confidence": "无",
        "reason": "",
        "indicators": {},
        "price_patterns": {}
    }

    # 获取最近的高低价收集
    lookback = min(8, len(df) - 1)  # 确保不超出数据范围
    highs = df['high'].tail(lookback).values
    lows = df['low'].tail(lookback).values
    closes = df['close'].tail(lookback).values

    print_colored(f"趋势分析 - 最近{len(closes)}个收盘价: {[round(x, 4) for x in closes]}", Colors.INFO)

    # 尝试获取指标，如果不存在则跳过相应的检查
    try:
        # 价格模式分析 - 修正高点低点比较逻辑
        # 检查是否形成更高的高点和更高的低点
        higher_highs = True
        higher_lows = True
        lower_highs = True
        lower_lows = True

        # 要求至少3个点才能形成趋势
        if len(highs) >= 3 and len(lows) >= 3:
            # 检查高点是否依次升高
            for i in range(2, len(highs)):
                if highs[i] <= highs[i - 1]:
                    higher_highs = False
                    break

            # 检查低点是否依次升高
            for i in range(2, len(lows)):
                if lows[i] <= lows[i - 1]:
                    higher_lows = False
                    break

            # 检查高点是否依次降低
            for i in range(2, len(highs)):
                if highs[i] >= highs[i - 1]:
                    lower_highs = False
                    break

            # 检查低点是否依次降低
            for i in range(2, len(lows)):
                if lows[i] >= lows[i - 1]:
                    lower_lows = False
                    break
        else:
            # 数据不足以判断趋势
            higher_highs = higher_lows = lower_highs = lower_lows = False

        trend_info["price_patterns"] = {
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows
        }

        price_pattern_text = (
            f"价格形态 - "
            f"高点走高: {format_log(str(higher_highs), Colors.GREEN if higher_highs else Colors.RED)}, "
            f"低点走高: {format_log(str(higher_lows), Colors.GREEN if higher_lows else Colors.RED)}, "
            f"高点走低: {format_log(str(lower_highs), Colors.GREEN if lower_highs else Colors.RED)}, "
            f"低点走低: {format_log(str(lower_lows), Colors.GREEN if lower_lows else Colors.RED)}"
        )
        print(price_pattern_text)

        # ===== 指标分析 =====
        # 1. 超级趋势分析
        if 'Supertrend_Direction' in df.columns:
            st_direction = df['Supertrend_Direction'].iloc[-1]
            st_consensus = df['Supertrend_Consensus'].iloc[-1] if 'Supertrend_Consensus' in df.columns else 0.0
            st_strength = df['Supertrend_Strength'].iloc[-1] if 'Supertrend_Strength' in df.columns else 0.0

            supertrend_trend = "UP" if st_direction > 0 else "DOWN" if st_direction < 0 else "NEUTRAL"

            # 保存到趋势信息字典
            trend_info["indicators"]["supertrend"] = {
                "trend": supertrend_trend,
                "consensus": float(st_consensus),
                "strength": float(st_strength)
            }

            print_colored(
                f"超级趋势方向: {Colors.GREEN if supertrend_trend == 'UP' else Colors.RED if supertrend_trend == 'DOWN' else Colors.GRAY}{supertrend_trend}{Colors.RESET}, "
                f"共识度: {st_consensus:.2f}, 强度: {st_strength:.2f}",
                Colors.INFO
            )
        else:
            supertrend_trend = "NEUTRAL"
            st_consensus = 0.0
            print_colored("未找到超级趋势指标", Colors.WARNING)
            trend_info["indicators"]["supertrend"] = {"trend": "NEUTRAL", "consensus": 0.0, "strength": 0.0}

        # 2. 威廉指标分析
        if 'Williams_R' in df.columns:
            williams_r = df['Williams_R'].iloc[-1]

            # 计算威廉指标的方向
            williams_direction = "flat"
            if len(df) >= 5:
                recent_williams = df['Williams_R'].tail(5).values
                williams_slope = np.polyfit(range(len(recent_williams)), recent_williams, 1)[0]

                if williams_slope > 1.5:
                    williams_direction = "strong_up"
                elif williams_slope > 0.5:
                    williams_direction = "up"
                elif williams_slope < -1.5:
                    williams_direction = "strong_down"
                elif williams_slope < -0.5:
                    williams_direction = "down"

            # 威廉指标的趋势判断
            if williams_r <= -80:
                williams_trend = "UP"  # 超卖区域，反转向上信号
                williams_state = "超卖"
            elif williams_r >= -20:
                williams_trend = "DOWN"  # 超买区域，反转向下信号
                williams_state = "超买"
            else:
                williams_trend = "NEUTRAL"
                williams_state = "中性"

            # 保存到趋势信息字典
            trend_info["indicators"]["williams"] = {
                "value": float(williams_r),
                "trend": williams_trend,
                "direction": williams_direction,
                "state": williams_state
            }

            print_colored(
                f"威廉指标: {Colors.GREEN if williams_r <= -80 else Colors.RED if williams_r >= -20 else Colors.RESET}{williams_r:.2f}{Colors.RESET} "
                f"({williams_state}), 趋势提示: {williams_trend}",
                Colors.GREEN if williams_trend == "UP" else Colors.RED if williams_trend == "DOWN" else Colors.RESET
            )
        else:
            williams_trend = "NEUTRAL"
            print_colored("未找到威廉指标", Colors.WARNING)
            trend_info["indicators"]["williams"] = {"value": -50, "trend": "NEUTRAL", "direction": "flat",
                                                    "state": "未知"}

        # 3. 其他指标分析
        # MACD
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            macd_cross = macd > macd_signal
            macd_trend = "UP" if macd_cross else "DOWN"

            # 保存到趋势信息字典
            trend_info["indicators"]["macd"] = {
                "value": float(macd),
                "signal": float(macd_signal),
                "trend": macd_trend,
                "histogram": float(macd - macd_signal)
            }

            print_colored(
                f"MACD趋势: {Colors.GREEN if macd_cross else Colors.RED}{macd_trend}{Colors.RESET}, "
                f"值: {macd:.6f}, 信号线: {macd_signal:.6f}, 差值: {macd - macd_signal:.6f}",
                Colors.INFO
            )
        else:
            macd_trend = "NEUTRAL"
            trend_info["indicators"]["macd"] = {"trend": "NEUTRAL", "value": 0, "signal": 0, "histogram": 0}

        # RSI
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            rsi_trend = "UP" if rsi > 55 else "DOWN" if rsi < 45 else "NEUTRAL"

            if rsi > 70:
                rsi_state = "超买"
            elif rsi < 30:
                rsi_state = "超卖"
            else:
                rsi_state = "中性"

            # 保存到趋势信息字典
            trend_info["indicators"]["rsi"] = {
                "value": float(rsi),
                "trend": rsi_trend,
                "state": rsi_state
            }

            print_colored(
                f"RSI: {Colors.RED if rsi > 70 else Colors.GREEN if rsi < 30 else Colors.RESET}{rsi:.2f}{Colors.RESET} "
                f"({rsi_state}), 趋势: {rsi_trend}",
                Colors.INFO
            )
        else:
            rsi_trend = "NEUTRAL"
            trend_info["indicators"]["rsi"] = {"value": 50, "trend": "NEUTRAL", "state": "未知"}

        # ===== 趋势综合判断 =====
        # 价格形态判断
        if higher_highs and higher_lows:
            price_trend = "UP"
        elif lower_highs and lower_lows:
            price_trend = "DOWN"
        else:
            price_trend = "NEUTRAL"

        trend_info["price_trend"] = price_trend

        # 综合多个指标判断趋势
        # 规则1: 超级趋势和价格形态一致时的高置信度判断
        if supertrend_trend == price_trend and supertrend_trend != "NEUTRAL":
            trend = supertrend_trend
            confidence = "高"
            reason = "超级趋势与价格形态一致"
            print_colored(f"趋势判断：{reason} ({trend})", Colors.GREEN if trend == "UP" else Colors.RED)

        # 规则2: 超级趋势有高共识度时的判断
        elif supertrend_trend != "NEUTRAL" and st_consensus >= 0.8:
            trend = supertrend_trend
            confidence = "中高"
            reason = "超级趋势共识度高"
            print_colored(f"趋势判断：{reason} ({trend})", Colors.GREEN if trend == "UP" else Colors.RED)

        # 规则3: 价格形态和威廉指标反向信号一致时的判断
        elif price_trend != "NEUTRAL" and williams_trend != "NEUTRAL" and price_trend != williams_trend:
            # 注意威廉指标超卖表示可能向上反转，所以与价格趋势相反时更有效
            trend = price_trend
            confidence = "中高"
            reason = "价格形态与威廉指标反转信号一致"
            print_colored(f"趋势判断：{reason} ({trend})", Colors.GREEN if trend == "UP" else Colors.RED)

        # 规则4: 价格形态明确时的判断
        elif price_trend != "NEUTRAL":
            trend = price_trend
            confidence = "中"
            reason = "价格形态明确"
            print_colored(f"趋势判断：{reason} ({trend})", Colors.GREEN if trend == "UP" else Colors.RED)

        # 规则5: 仅有超级趋势方向时的判断
        elif supertrend_trend != "NEUTRAL":
            trend = supertrend_trend
            confidence = "低"
            reason = "仅超级趋势有方向"
            print_colored(f"趋势判断：{reason} ({trend})", Colors.YELLOW)

        # 规则6: 威廉指标与RSI形成背离的反转信号
        elif williams_trend != "NEUTRAL" and (
                (williams_trend == "UP" and rsi < 40) or
                (williams_trend == "DOWN" and rsi > 60)
        ):
            trend = williams_trend
            confidence = "低"
            reason = "威廉指标与RSI形成背离，可能是反转信号"
            print_colored(f"趋势判断：{reason} ({trend})", Colors.YELLOW)

        # 规则7: 无法确定明确趋势
        else:
            trend = "NEUTRAL"
            confidence = "无"
            reason = "无法确定明确趋势"
            print_colored(f"趋势判断：{reason}", Colors.GRAY)

        # 更新趋势信息
        trend_info["trend"] = trend
        trend_info["confidence"] = confidence
        trend_info["reason"] = reason

        # 使用ADX确认趋势强度
        if 'ADX' in df.columns:
            adx = df['ADX'].iloc[-1]
            trend_info["indicators"]["adx"] = float(adx)

            if adx < 20 and trend != "NEUTRAL":
                print_colored(f"ADX低 ({adx:.2f} < 20)，趋势较弱", Colors.YELLOW)
                if confidence == "低":
                    trend = "NEUTRAL"
                    confidence = "无"
                    reason += "，ADX低确认趋势弱"
                    print_colored(f"由于ADX低且趋势置信度低，修正为中性趋势", Colors.YELLOW)

                    # 更新趋势信息
                    trend_info["trend"] = trend
                    trend_info["confidence"] = confidence
                    trend_info["reason"] = reason
            elif adx >= 25:
                print_colored(f"ADX高 ({adx:.2f} >= 25)，趋势强劲", Colors.GREEN)
                if confidence in ["中", "低"]:
                    confidence = "中高"
                    reason += "，ADX高确认趋势强"

                    # 更新趋势信息
                    trend_info["confidence"] = confidence
                    trend_info["reason"] = reason

        # 打印最终趋势判断
        trend_color = Colors.GREEN if trend == "UP" else Colors.RED if trend == "DOWN" else Colors.GRAY
        confidence_color = (Colors.GREEN if confidence == "高" or confidence == "中高" else
                            Colors.YELLOW if confidence == "中" else
                            Colors.RED if confidence == "低" else Colors.GRAY)

        print_colored(
            f"最终趋势判断: {trend_color}{trend}{Colors.RESET}, "
            f"置信度: {confidence_color}{confidence}{Colors.RESET}, "
            f"原因: {reason}",
            Colors.BOLD
        )

        # 计算趋势持续时间
        duration = 0
        if trend == "UP":
            for i in range(2, len(df)):
                if not (df['high'].iloc[-i] >= df['high'].iloc[-i - 1] or df['low'].iloc[-i] >= df['low'].iloc[-i - 1]):
                    break
                duration += 1
        elif trend == "DOWN":
            for i in range(2, len(df)):
                if not (df['high'].iloc[-i] <= df['high'].iloc[-i - 1] or df['low'].iloc[-i] <= df['low'].iloc[-i - 1]):
                    break
                duration += 1

        # 转换为分钟
        candle_minutes = 15  # 假设15分钟K线
        duration = duration * candle_minutes
        duration_hours = duration / 60
        duration_text = f"{duration}分钟" if duration_hours < 1 else f"{duration_hours:.1f}小时"

        print_colored(f"趋势持续时间: {duration_text}", Colors.INFO)

        # 限制最大持续时间
        duration = min(duration, config["TREND_DURATION_THRESHOLD"])

        # 更新趋势信息
        trend_info["duration"] = duration
        trend_info["duration_minutes"] = duration

        if logger:
            logger.info("SMC 趋势分析", extra={
                "trend": trend,
                "duration": duration,
                "confidence": confidence,
                "reason": reason,
                "supertrend": supertrend_trend,
                "price_trend": price_trend,
                "williams_trend": williams_trend,
                "adx": adx if 'ADX' in df.columns else None
            })

            return trend, duration, trend_info


    def detect_order_blocks_3d(df, volume_threshold=1.3, price_deviation=0.002, consolidation_bars=3):
        """
        三维订单块检测：成交量+价格波动+震荡验证
        参数：
            volume_threshold: 成交量倍数阈值
            price_deviation: 最大允许价格波动（ATR比率）
            consolidation_bars: 震荡验证所需K线数
        """
        order_blocks = []
        atr = df['ATR'].values

        for i in range(1, len(df)):
            # 成交量激增检测
            vol_ratio = df['volume'].iloc[i] / df['volume'].iloc[i - 3:i].mean()

            # 价格波动检测
            price_change = abs(df['close'].iloc[i] - df['close'].iloc[i - 1])
            atr_ratio = price_change / atr[i] if atr[i] > 0 else 0

            # 震荡验证
            is_consolidation = all(
                abs(df['high'].iloc[j] - df['low'].iloc[j]) < 0.5 * atr[j]
                for j in range(i - consolidation_bars + 1, i + 1)
            )

            if (vol_ratio > volume_threshold and
                    atr_ratio < price_deviation and
                    is_consolidation):
                block_type = "bid" if df['close'].iloc[i] > df['open'].iloc[i] else "ask"
                order_blocks.append({
                    'index': i,
                    'price': df['close'].iloc[i],
                    'type': block_type,
                    'strength': vol_ratio * (1 - atr_ratio)
                })

        # 趋势过滤：仅保留与当前趋势同向的订单块
        trend = get_smc_trend(df)
        return [b for b in order_blocks if
                (trend == 'up' and b['type'] == 'bid') or
                (trend == 'down' and b['type'] == 'ask')]


        except Exception as e:
        print_colored(f"❌ 趋势分析出错: {e}", Colors.ERROR)
        if logger:
            logger.error(f"趋势分析出错: {e}")
        return "NEUTRAL", 0, {"confidence": "无", "reason": f"分析出错: {str(e)}"}


def find_swing_points(df: pd.DataFrame, window=3):
    """
    改进摆动点识别，增加窗口参数以平滑噪声

    参数:
        df: 包含OHLC数据的DataFrame
        window: 寻找摆动点的窗口大小

    返回:
        swing_highs: 摆动高点列表
        swing_lows: 摆动低点列表
    """
    swing_highs = []
    swing_lows = []

    if len(df) <= 2 * window:
        indicators_logger.warning(f"数据长度 {len(df)} 不足以找到摆动点 (需要 > {2 * window})")
        print_colored(f"⚠️ 数据长度 {len(df)} 不足以找到摆动点", Colors.WARNING)
        return swing_highs, swing_lows

    try:
        for i in range(window, len(df) - window):
            # 摆动高点：当前高点大于前后window根K线的高点
            if all(df['high'].iloc[i] > df['high'].iloc[j] for j in range(i - window, i)) and \
                    all(df['high'].iloc[i] > df['high'].iloc[j] for j in range(i + 1, i + window + 1)):
                swing_highs.append(df['high'].iloc[i])
                print_colored(f"发现摆动高点: 索引={i}, 价格={df['high'].iloc[i]:.4f}", Colors.INFO)

            # 摆动低点：当前低点小于前后window根K线的低点
            if all(df['low'].iloc[i] < df['low'].iloc[j] for j in range(i - window, i)) and \
                    all(df['low'].iloc[i] < df['low'].iloc[j] for j in range(i + 1, i + window + 1)):
                swing_lows.append(df['low'].iloc[i])
                print_colored(f"发现摆动低点: 索引={i}, 价格={df['low'].iloc[i]:.4f}", Colors.INFO)

        # 如果没有找到任何摆动点，使用简化的算法
        if not swing_highs or not swing_lows:
            print_colored("使用简化算法寻找摆动点", Colors.INFO)
            window = max(2, window // 2)  # 缩小窗口

            for i in range(window, len(df) - window):
                # 简化版摆动高点
                if df['high'].iloc[i] == max(df['high'].iloc[i - window:i + window + 1]):
                    swing_highs.append(df['high'].iloc[i])
                    print_colored(f"简化算法发现高点: 索引={i}, 价格={df['high'].iloc[i]:.4f}", Colors.INFO)

                # 简化版摆动低点
                if df['low'].iloc[i] == min(df['low'].iloc[i - window:i + window + 1]):
                    swing_lows.append(df['low'].iloc[i])
                    print_colored(f"简化算法发现低点: 索引={i}, 价格={df['low'].iloc[i]:.4f}", Colors.INFO)

        print_colored(f"找到 {len(swing_highs)} 个摆动高点和 {len(swing_lows)} 个摆动低点", Colors.INFO)
        return swing_highs, swing_lows
    except Exception as e:
        indicators_logger.error(f"寻找摆动点失败: {e}")
        print_colored(f"❌ 寻找摆动点失败: {e}", Colors.ERROR)
        return [], []


def calculate_fibonacci_retracements(df: pd.DataFrame):
    """
    改进斐波那契回撤计算

    参数:
        df: 包含OHLC数据的DataFrame

    返回:
        fib_levels: 斐波那契回撤水平列表
    """
    try:
        # 获取摆动点
        swing_highs, swing_lows = find_swing_points(df)

        # 如果没有足够的摆动点，返回当前价格作为默认值
        if not swing_highs or not swing_lows:
            indicators_logger.warning("无法计算斐波那契回撤：无有效的摆动高点或低点")
            print_colored("⚠️ 无法计算斐波那契回撤：无有效的摆动高点或低点", Colors.WARNING)
            return [df['close'].iloc[-1]] * 5

        # 确定趋势方向
        current_close = df['close'].iloc[-1]
        avg_high = sum(swing_highs[-min(3, len(swing_highs)):]) / min(3, len(swing_highs))
        avg_low = sum(swing_lows[-min(3, len(swing_lows)):]) / min(3, len(swing_lows))

        print_colored(f"当前价格: {current_close:.4f}, 平均高点: {avg_high:.4f}, 平均低点: {avg_low:.4f}", Colors.INFO)

        # 确定A和B点 (趋势高低点)
        if current_close > avg_high:  # 上升趋势，从最低点到最高点
            A = min(swing_lows) if swing_lows else df['low'].min()
            B = max(swing_highs) if swing_highs else df['high'].max()
            print_colored(f"上升趋势斐波那契: 最低点={A:.4f}, 最高点={B:.4f}", Colors.INFO)
        elif current_close < avg_low:  # 下降趋势，从最高点到最低点
            A = max(swing_highs) if swing_highs else df['high'].max()
            B = min(swing_lows) if swing_lows else df['low'].min()
            print_colored(f"下降趋势斐波那契: 最高点={A:.4f}, 最低点={B:.4f}", Colors.INFO)
        else:  # 使用最近的波动
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                recent_high = max(swing_highs[-2:])
                recent_low = min(swing_lows[-2:])
                if recent_high > recent_low:
                    A = recent_low
                    B = recent_high
                    print_colored(f"短期上升波动: 低点={A:.4f}, 高点={B:.4f}", Colors.INFO)
                else:
                    A = recent_high
                    B = recent_low
                    print_colored(f"短期下降波动: 高点={A:.4f}, 低点={B:.4f}", Colors.INFO)
            else:
                # 使用最大最小值
                A = df['low'].min()
                B = df['high'].max()
                print_colored(f"使用全局极值: 最低={A:.4f}, 最高={B:.4f}", Colors.INFO)

        # 确保A < B用于一致的计算方向
        is_reversed = False
        if A > B:
            A, B = B, A
            is_reversed = True
            print_colored("调整计算方向", Colors.INFO)

        # 确保点不重合
        if abs(B - A) < df['ATR'].iloc[-1] * 0.1 if 'ATR' in df.columns else 0.001:
            indicators_logger.warning("斐波那契点过于接近，扩大范围")
            print_colored("⚠️ 斐波那契点过于接近，扩大范围", Colors.WARNING)
            A = A * 0.99
            B = B * 1.01

        # 计算斐波那契水平
        retracements = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]  # 增加更多水平
        fib_levels = [A + (B - A) * retr for retr in retracements]

        # 如果是反向的，还原顺序以保持一致性
        if is_reversed:
            fib_levels.reverse()
            print_colored("反转斐波那契水平顺序", Colors.INFO)

        print_colored(f"斐波那契水平: {[round(level, 4) for level in fib_levels]}", Colors.INFO)
        return fib_levels
    except Exception as e:
        indicators_logger.error(f"计算斐波那契回撤失败: {e}")
        print_colored(f"❌ 计算斐波那契回撤失败: {e}", Colors.ERROR)
        # 返回当前价格附近的默认值
        current_price = df['close'].iloc[-1]
        return [current_price * (1 - 0.05 + i * 0.02) for i in range(5)]


def calculate_optimized_indicators(df: pd.DataFrame, btc_df=None):
    """
    计算优化后的指标，允许部分指标因数据不足而跳过，但关键指标缺失时返回空DataFrame
    增强版：优化超级趋势计算和提供更多日志信息

    参数:
        df: 包含OHLC数据的DataFrame
        btc_df: BTC价格数据，用于计算整体市场情绪

    返回:
        df: 添加了各种指标的DataFrame
    """
    try:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        critical_indicators = ['RSI', 'MACD', 'EMA5', 'EMA20']
        all_indicators = ['VWAP', 'EMA24', 'EMA52', 'MACD', 'MACD_signal', 'RSI', 'OBV', 'TR',
                          'ATR', 'Momentum', 'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower',
                          'ROC', 'ADX', 'Market_Sentiment', 'CCI', 'EMA5', 'EMA20', 'Panic_Index',
                          'Supertrend', 'Supertrend_Direction', 'SMMA60', 'Williams_R']

        # 检查输入数据
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            print_colored(
                f"⚠️ 输入数据无效或缺失必要列: {[col for col in required_cols if col not in df.columns]}",
                Colors.WARNING
            )
            indicators_logger.info(f"输入数据无效或缺失列（{required_cols}）")
            return pd.DataFrame()

        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 数据概览
        print_colored(
            f"处理数据: {len(df)}行, 收盘价范围: {df['close'].min():.4f} - {df['close'].max():.4f}",
            Colors.INFO
        )

        if df['close'].sum() == 0:
            print_colored("❌ 数据无效：收盘价全为0", Colors.ERROR)
            indicators_logger.info("数据无效：close 列全为 0")
            return pd.DataFrame()

        # 初始化指标列
        for col in all_indicators:
            if col not in df.columns:
                df[col] = np.nan

        # 计算VWAP
        if len(df) >= 50:
            df['VWAP'] = (df['close'] * df['volume']).rolling(window=50, min_periods=1).sum() / \
                         df['volume'].rolling(window=50, min_periods=1).sum().replace(0, np.finfo(float).eps)
            log_indicator(None, "VWAP", df['VWAP'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算VWAP（需要50根K线）", Colors.WARNING)

        # 计算各种EMA和MACD
        if len(df) >= 5:
            df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
            log_indicator(None, "EMA5", df['EMA5'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算EMA5（需要5根K线）", Colors.WARNING)

        if len(df) >= 20:
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            log_indicator(None, "EMA20", df['EMA20'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算EMA20（需要20根K线）", Colors.WARNING)

        if len(df) >= 24:
            df['EMA24'] = df['close'].ewm(span=24, adjust=False).mean()
            log_indicator(None, "EMA24", df['EMA24'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算EMA24（需要24根K线）", Colors.WARNING)

        if len(df) >= 52:
            df['EMA52'] = df['close'].ewm(span=52, adjust=False).mean()
            log_indicator(None, "EMA52", df['EMA52'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算EMA52（需要52根K线）", Colors.WARNING)

        # 计算MACD
        if len(df) >= 26:  # 减少所需数据点
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

            macd_color = Colors.GREEN if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else Colors.RED
            print_colored(
                f"MACD 计算完成，最后值: {macd_color}{df['MACD'].iloc[-1]:.4f}{Colors.RESET}, "
                f"信号线: {df['MACD_signal'].iloc[-1]:.4f}, "
                f"柱状图: {df['MACD_histogram'].iloc[-1]:.4f}",
                Colors.INFO
            )
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算MACD（需要26根K线）", Colors.WARNING)

        # 计算RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
            loss = -delta.clip(upper=0).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)
            df['RSI'] = 100 - (100 / (1 + rs))

            rsi_value = df['RSI'].iloc[-1]
            rsi_color = Colors.RED if rsi_value > 70 else Colors.GREEN if rsi_value < 30 else Colors.RESET
            print_colored(f"RSI 计算完成，最后值: {rsi_color}{rsi_value:.2f}{Colors.RESET}", Colors.INFO)
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算RSI（需要14根K线）", Colors.WARNING)

        # 计算威廉指标
        if len(df) >= 14:
            df = calculate_williams_r(df, period=14)
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算Williams %R（需要14根K线）", Colors.WARNING)

        # 计算OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        log_indicator(None, "OBV", df['OBV'].iloc[-1])

        # 计算ATR
        if len(df) >= 14:
            df['TR'] = np.maximum(df['high'] - df['low'],
                                  np.maximum(abs(df['high'] - df['close'].shift(1)),
                                             abs(df['low'] - df['close'].shift(1))))
            df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
            log_indicator(None, "ATR", df['ATR'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算ATR（需要14根K线）", Colors.WARNING)

        # 计算动量
        if len(df) >= 10:
            df['Momentum'] = df['close'] - df['close'].shift(10)
            log_indicator(None, "Momentum", df['Momentum'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算Momentum（需要10根K线）", Colors.WARNING)

        # 计算布林带
        if len(df) >= 20:
            df['BB_Middle'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['BB_Std'] = df['close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

            # 计算价格相对布林带位置
            bb_position = (df['close'].iloc[-1] - df['BB_Lower'].iloc[-1]) / (
                    df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])
            bb_position_text = (
                "上轨以上" if bb_position > 1 else
                "上轨附近" if bb_position > 0.9 else
                "上轨和中轨之间" if bb_position > 0.5 else
                "中轨附近" if bb_position > 0.45 and bb_position < 0.55 else
                "中轨和下轨之间" if bb_position > 0.1 else
                "下轨附近" if bb_position > 0 else
                "下轨以下"
            )

            bb_position_color = (
                Colors.RED if bb_position > 0.9 else
                Colors.YELLOW if bb_position > 0.7 else
                Colors.GREEN if bb_position < 0.3 else
                Colors.RESET
            )

            print_colored(
                f"布林带计算完成 - 上轨: {df['BB_Upper'].iloc[-1]:.4f}, "
                f"中轨: {df['BB_Middle'].iloc[-1]:.4f}, "
                f"下轨: {df['BB_Lower'].iloc[-1]:.4f}",
                Colors.INFO
            )
            print_colored(
                f"价格在布林带的位置: {bb_position_color}{bb_position:.2f} ({bb_position_text}){Colors.RESET}",
                Colors.INFO
            )
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算Bollinger Bands（需要20根K线）", Colors.WARNING)

        # 计算变化率
        if len(df) >= 5:
            df['ROC'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5).replace(0,
                                                                                            np.finfo(float).eps) * 100
            log_indicator(None, "ROC", df['ROC'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算ROC（需要5根K线）", Colors.WARNING)

        # 计算ADX
        if len(df) >= 14:
            df['Plus_DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
            df['Minus_DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
            df['TR14'] = df['TR'].rolling(window=14, min_periods=1).sum()
            df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(window=14, min_periods=1).sum() / df['TR14'].replace(0,
                                                                                                              np.finfo(
                                                                                                                  float).eps))
            df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(window=14, min_periods=1).sum() / df['TR14'].replace(0,
                                                                                                                np.finfo(
                                                                                                                    float).eps))
            df['DX'] = 100 * abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI']).replace(0, np.finfo(
                float).eps)
            df['ADX'] = df['DX'].rolling(window=14, min_periods=1).mean()

            adx_value = df['ADX'].iloc[-1]
            adx_strength = (
                "强烈趋势" if adx_value >= 35 else
                "趋势" if adx_value >= 25 else
                "弱趋势" if adx_value >= 20 else
                "无趋势"
            )
            adx_color = (
                Colors.GREEN + Colors.BOLD if adx_value >= 35 else
                Colors.GREEN if adx_value >= 25 else
                Colors.YELLOW if adx_value >= 20 else
                Colors.GRAY
            )

            print_colored(f"ADX 计算完成，最后值: {adx_color}{adx_value:.2f} ({adx_strength}){Colors.RESET}",
                          Colors.INFO)
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算ADX（需要14根K线）", Colors.WARNING)

        # 计算CCI
        if len(df) >= 20:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
            mean_dev = (typical_price - sma_tp).abs().rolling(window=20, min_periods=1).mean()
            df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.finfo(float).eps))

            cci_value = df['CCI'].iloc[-1]
            cci_color = Colors.RED if cci_value > 100 else Colors.GREEN if cci_value < -100 else Colors.RESET
            cci_state = "超买" if cci_value > 100 else "超卖" if cci_value < -100 else "中性"

            print_colored(f"CCI 计算完成，最后值: {cci_color}{cci_value:.2f} ({cci_state}){Colors.RESET}", Colors.INFO)
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算CCI（需要20根K线）", Colors.WARNING)

        # 计算超级趋势指标
        if len(df) >= 14 and 'ATR' in df.columns:
            df = calculate_supertrend(df)
        else:
            print_colored(f"⚠️ 数据不足或缺失ATR，无法计算Supertrend", Colors.WARNING)

        # 计算SMMA
        if len(df) >= 60:
            df = calculate_smma(df, period=60)
            log_indicator(None, "SMMA60", df['SMMA60'].iloc[-1])
        else:
            print_colored(f"⚠️ 数据不足（{len(df)}根K线），无法计算SMMA60（需要60根K线）", Colors.WARNING)

        # 计算市场情绪和恐慌指数
        has_btc_data = btc_df is not None and not btc_df.empty and len(btc_df) >= 6
        if has_btc_data:
            btc_change = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-6]) / btc_df['close'].iloc[-6] * 100
            print_colored(f"BTC变化率: {Colors.GREEN if btc_change > 0 else Colors.RED}{btc_change:.2f}%{Colors.RESET}",
                          Colors.INFO)

            if btc_change > 2.0:
                df['Market_Sentiment'] = 1  # 强烈看多
                sentiment_desc = "强烈看多"
                sentiment_color = Colors.GREEN + Colors.BOLD
            elif btc_change > 1.0:
                df['Market_Sentiment'] = 0.8  # 看多
                sentiment_desc = "看多"
                sentiment_color = Colors.GREEN
            elif btc_change > 0.2:
                df['Market_Sentiment'] = 0.6  # 轻微看多
                sentiment_desc = "轻微看多"
                sentiment_color = Colors.GREEN
            elif btc_change < -2.0:
                df['Market_Sentiment'] = -1  # 强烈看空
                sentiment_desc = "强烈看空"
                sentiment_color = Colors.RED + Colors.BOLD
            elif btc_change < -1.0:
                df['Market_Sentiment'] = -0.8  # 看空
                sentiment_desc = "看空"
                sentiment_color = Colors.RED
            elif btc_change < -0.2:
                df['Market_Sentiment'] = -0.6  # 轻微看空
                sentiment_desc = "轻微看空"
                sentiment_color = Colors.RED
            else:
                df['Market_Sentiment'] = 0  # 中性
                sentiment_desc = "中性"
                sentiment_color = Colors.RESET

            print_colored(
                f"市场情绪: {sentiment_color}{sentiment_desc}{Colors.RESET} ({df['Market_Sentiment'].iloc[-1]:.1f})",
                Colors.INFO)

            # 计算恐慌指数 - 考虑BTC波动和当前ATR
            if 'ATR' in df.columns:
                atr_mean = df['ATR'].mean()
                atr_ratio = df['ATR'].iloc[-1] / atr_mean if atr_mean != 0 else 1

                # 综合BTC波动和ATR比率计算恐慌指数
                btc_factor = abs(btc_change) / 2  # BTC波动贡献
                atr_factor = (atr_ratio - 1) * 5 if atr_ratio > 1 else 0  # ATR贡献

                panic_index = min(10, max(0, 5 + btc_factor + atr_factor))
                df['Panic_Index'] = panic_index

                panic_color = (
                    Colors.RED + Colors.BOLD if panic_index > 7 else
                    Colors.RED if panic_index > 5 else
                    Colors.YELLOW if panic_index > 3 else
                    Colors.GREEN
                )

                panic_level = (
                    "极度恐慌" if panic_index > 7 else
                    "恐慌" if panic_index > 5 else
                    "谨慎" if panic_index > 3 else
                    "平静"
                )

                print_colored(
                    f"恐慌指数: {panic_color}{panic_index:.2f}/10 ({panic_level}){Colors.RESET} "
                    f"[BTC波动:{btc_factor:.1f}, ATR比率:{atr_ratio:.2f}]",
                    Colors.INFO
                )
            else:
                df['Panic_Index'] = 5  # 默认中等恐慌
                print_colored(f"恐慌指数: 5.00/10 (默认值，无ATR数据)", Colors.INFO)
        else:
            # 仅使用ATR计算恐慌指数
            if 'ATR' in df.columns:
                atr_mean = df['ATR'].rolling(window=20).mean().iloc[-1]
                atr_ratio = df['ATR'].iloc[-1] / atr_mean if atr_mean != 0 else 1

                panic_index = min(10, (1 + (atr_ratio - 1) * 5)) if atr_ratio > 1 else 3
                df['Market_Sentiment'] = 0  # 无BTC数据，默认中性
                df['Panic_Index'] = panic_index

                panic_color = (
                    Colors.RED + Colors.BOLD if panic_index > 7 else
                    Colors.RED if panic_index > 5 else
                    Colors.YELLOW if panic_index > 3 else
                    Colors.GREEN
                )

                panic_level = (
                    "极度恐慌" if panic_index > 7 else
                    "恐慌" if panic_index > 5 else
                    "谨慎" if panic_index > 3 else
                    "平静"
                )

                print_colored(
                    f"市场情绪: 中性 (0.0，无BTC数据)",
                    Colors.INFO
                )
                print_colored(
                    f"恐慌指数: {panic_color}{panic_index:.2f}/10 ({panic_level}){Colors.RESET} "
                    f"[仅基于ATR比率:{atr_ratio:.2f}]",
                    Colors.INFO
                )
            else:
                df['Market_Sentiment'] = 0
                df['Panic_Index'] = 5
                print_colored(f"市场情绪: 中性 (0.0，无BTC数据)", Colors.INFO)
                print_colored(f"恐慌指数: 5.00/10 (默认值，无ATR和BTC数据)", Colors.INFO)

            # 检查关键指标是否计算成功
        missing_critical = [indicator for indicator in critical_indicators if
                            indicator not in df.columns or df[indicator].isna().all()]
        if missing_critical:
            print_colored(f"❌ 关键指标计算失败: {missing_critical}", Colors.ERROR)
            indicators_logger.error(f"关键指标 {missing_critical} 计算失败，停止计算")
            return pd.DataFrame()

        # 填充缺失的指标
        for col in all_indicators:
            if col not in df.columns or df[col].isna().all():
                df[col] = 0.0
                indicators_logger.warning(f"{col} 计算失败，填充默认值")

        print_colored(f"✅ 所有指标计算完成，总计 {len(all_indicators)} 个指标", Colors.GREEN + Colors.BOLD)
        return df
    except Exception as e:
        print_colored(f"❌ 计算优化指标失败: {e}", Colors.ERROR)
        indicators_logger.error(f"计算优化指标失败: {e}")
        return pd.DataFrame()