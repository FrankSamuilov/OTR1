"""
高级指标模块
包含随机动量指数(SMI)、随机指标(Stochastic Oscillator)和抛物线转向指标(Parabolic SAR)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from logger_utils import Colors, print_colored


def calculate_smi(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_period: int = 3) -> pd.DataFrame:
    """
    计算随机动量指数 (Stochastic Momentum Index)

    参数:
        df: 价格数据DataFrame
        k_period: K周期，默认14
        d_period: D周期，默认3
        smooth_period: 平滑周期，默认3

    返回:
        df: 添加了SMI指标的DataFrame
    """
    try:
        if len(df) < k_period:
            print_colored(f"⚠️ 数据长度 {len(df)} 小于SMI周期 {k_period}", Colors.WARNING)
            return df

        # 获取收盘价
        close = df['close']

        # 计算各周期最高价和最低价
        high_k = df['high'].rolling(window=k_period).max()
        low_k = df['low'].rolling(window=k_period).min()

        # 计算中心点
        mid_point = (high_k + low_k) / 2

        # 计算价格位置
        distance = close - mid_point

        # 计算价格范围
        value_range = (high_k - low_k) / 2

        # 平滑计算
        smoothed_distance = distance.ewm(span=smooth_period).mean()
        smoothed_range = value_range.ewm(span=smooth_period).mean()

        # 计算SMI
        smi = 100 * (smoothed_distance / smoothed_range.replace(0, np.finfo(float).eps))

        # 计算信号线
        smi_signal = smi.ewm(span=d_period).mean()

        # 添加到DataFrame
        df['SMI'] = smi
        df['SMI_Signal'] = smi_signal
        df['SMI_Histogram'] = smi - smi_signal

        # 打印最新的SMI值
        last_smi = df['SMI'].iloc[-1]
        last_signal = df['SMI_Signal'].iloc[-1]
        last_hist = df['SMI_Histogram'].iloc[-1]

        smi_color = (
            Colors.GREEN if last_smi > 40 else
            Colors.RED if last_smi < -40 else
            Colors.RESET
        )

        print_colored(
            f"SMI: {smi_color}{last_smi:.2f}{Colors.RESET}, "
            f"信号线: {last_signal:.2f}, "
            f"直方图: {last_hist:.2f}",
            Colors.INFO
        )

        return df
    except Exception as e:
        print_colored(f"❌ 计算SMI指标失败: {e}", Colors.ERROR)
        return df


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    """
    计算随机指标 (Stochastic Oscillator)

    参数:
        df: 价格数据DataFrame
        k_period: K周期，默认14
        d_period: D周期，默认3
        slowing: 慢化因子，默认3

    返回:
        df: 添加了随机指标的DataFrame
    """
    try:
        if len(df) < k_period:
            print_colored(f"⚠️ 数据长度 {len(df)} 小于随机指标周期 {k_period}", Colors.WARNING)
            return df

        # 计算N日内的最高价和最低价
        high_k = df['high'].rolling(window=k_period).max()
        low_k = df['low'].rolling(window=k_period).min()

        # 计算未经平滑的%K
        k_raw = 100 * (df['close'] - low_k) / (high_k - low_k).replace(0, np.finfo(float).eps)

        # 应用慢化因子
        k_slowed = k_raw.rolling(window=slowing).mean()

        # 计算%D
        d = k_slowed.rolling(window=d_period).mean()

        # 添加到DataFrame
        df['Stochastic_K'] = k_slowed
        df['Stochastic_D'] = d

        # 添加交叉信号
        df['Stochastic_Cross_Up'] = (
                (df['Stochastic_K'] > df['Stochastic_D']) &
                (df['Stochastic_K'].shift(1) <= df['Stochastic_D'].shift(1))
        ).astype(int)

        df['Stochastic_Cross_Down'] = (
                (df['Stochastic_K'] < df['Stochastic_D']) &
                (df['Stochastic_K'].shift(1) >= df['Stochastic_D'].shift(1))
        ).astype(int)

        # 打印最新的随机指标值
        last_k = df['Stochastic_K'].iloc[-1]
        last_d = df['Stochastic_D'].iloc[-1]

        # 确定状态
        if last_k > 80 and last_d > 80:
            state = "超买"
            color = Colors.OVERBOUGHT
        elif last_k < 20 and last_d < 20:
            state = "超卖"
            color = Colors.OVERSOLD
        else:
            state = "中性"
            color = Colors.RESET

        # 检查交叉信号
        cross_up = df['Stochastic_Cross_Up'].iloc[-1]
        cross_down = df['Stochastic_Cross_Down'].iloc[-1]

        cross_message = ""
        if cross_up:
            cross_message = f"{Colors.GREEN}K线上穿D线{Colors.RESET}"
        elif cross_down:
            cross_message = f"{Colors.RED}K线下穿D线{Colors.RESET}"

        print_colored(
            f"随机指标: {color}K({last_k:.2f}) D({last_d:.2f}){Colors.RESET} - {state} {cross_message}",
            Colors.INFO
        )

        return df
    except Exception as e:
        print_colored(f"❌ 计算随机指标失败: {e}", Colors.ERROR)
        return df


def calculate_parabolic_sar(df: pd.DataFrame, initial_af: float = 0.02, max_af: float = 0.2,
                            af_step: float = 0.02) -> pd.DataFrame:
    """
    计算抛物线转向指标 (Parabolic SAR)

    参数:
        df: 价格数据DataFrame
        initial_af: 初始加速因子，默认0.02
        max_af: 最大加速因子，默认0.2
        af_step: 加速因子步长，默认0.02

    返回:
        df: 添加了SAR指标的DataFrame
    """
    try:
        if len(df) < 5:  # 至少需要几个数据点
            print_colored(f"⚠️ 数据长度 {len(df)} 不足以计算SAR", Colors.WARNING)
            return df

        # 初始化SAR列
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))  # 1表示上升趋势，-1表示下降趋势
        extreme_point = np.zeros(len(df))
        acceleration_factor = np.zeros(len(df))

        # 初始化第一个点
        # 简单起见，假设第一个点是上升趋势
        trend[0] = 1
        sar[0] = df['low'].iloc[0]
        extreme_point[0] = df['high'].iloc[0]
        acceleration_factor[0] = initial_af

        # 计算后续点
        for i in range(1, len(df)):
            # 如果上一个周期是上升趋势
            if trend[i - 1] == 1:
                # SAR值计算
                sar[i] = sar[i - 1] + acceleration_factor[i - 1] * (extreme_point[i - 1] - sar[i - 1])

                # 确保SAR不高于前两个周期的最低价
                if i >= 2:
                    sar[i] = min(sar[i], min(df['low'].iloc[i - 1], df['low'].iloc[i - 2]))

                # 如果价格跌破SAR，转为下降趋势
                if df['low'].iloc[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = extreme_point[i - 1]
                    extreme_point[i] = df['low'].iloc[i]
                    acceleration_factor[i] = initial_af
                else:
                    # 继续上升趋势
                    trend[i] = 1

                    # 更新极值点
                    if df['high'].iloc[i] > extreme_point[i - 1]:
                        extreme_point[i] = df['high'].iloc[i]
                        # 更新加速因子
                        acceleration_factor[i] = min(max_af, acceleration_factor[i - 1] + af_step)
                    else:
                        extreme_point[i] = extreme_point[i - 1]
                        acceleration_factor[i] = acceleration_factor[i - 1]

            # 如果上一个周期是下降趋势
            else:
                # SAR值计算
                sar[i] = sar[i - 1] + acceleration_factor[i - 1] * (extreme_point[i - 1] - sar[i - 1])

                # 确保SAR不低于前两个周期的最高价
                if i >= 2:
                    sar[i] = max(sar[i], max(df['high'].iloc[i - 1], df['high'].iloc[i - 2]))

                # 如果价格突破SAR，转为上升趋势
                if df['high'].iloc[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = extreme_point[i - 1]
                    extreme_point[i] = df['high'].iloc[i]
                    acceleration_factor[i] = initial_af
                else:
                    # 继续下降趋势
                    trend[i] = -1

                    # 更新极值点
                    if df['low'].iloc[i] < extreme_point[i - 1]:
                        extreme_point[i] = df['low'].iloc[i]
                        # 更新加速因子
                        acceleration_factor[i] = min(max_af, acceleration_factor[i - 1] + af_step)
                    else:
                        extreme_point[i] = extreme_point[i - 1]
                        acceleration_factor[i] = acceleration_factor[i - 1]

        # 添加到DataFrame
        df['SAR'] = sar
        df['SAR_Trend'] = trend

        # 检查趋势变化
        df['SAR_Trend_Change'] = df['SAR_Trend'].diff().fillna(0).abs()

        # 打印最新的SAR值
        last_sar = df['SAR'].iloc[-1]
        last_price = df['close'].iloc[-1]
        last_trend = "上升" if df['SAR_Trend'].iloc[-1] == 1 else "下降"
        trend_change = df['SAR_Trend_Change'].iloc[-1] > 0

        # 计算与价格的距离
        distance = abs(last_price - last_sar) / last_price * 100  # 百分比

        trend_color = Colors.GREEN if df['SAR_Trend'].iloc[-1] == 1 else Colors.RED

        print_colored(
            f"抛物线SAR: {last_sar:.6f}, 趋势: {trend_color}{last_trend}{Colors.RESET}, "
            f"距离价格: {distance:.2f}%, {'⚠️ 刚刚转向' if trend_change else ''}",
            Colors.INFO
        )

        return df
    except Exception as e:
        print_colored(f"❌ 计算SAR指标失败: {e}", Colors.ERROR)
        return df


def analyze_advanced_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    综合分析高级指标，生成交易信号和评估质量

    参数:
        df: 包含计算好的高级指标的DataFrame

    返回:
        包含分析结果的字典
    """
    try:
        result = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "reasons": [],
            "indicators": {}
        }

        # 检查各指标是否存在
        has_smi = all(col in df.columns for col in ['SMI', 'SMI_Signal'])
        has_stochastic = all(col in df.columns for col in ['Stochastic_K', 'Stochastic_D'])
        has_sar = all(col in df.columns for col in ['SAR', 'SAR_Trend'])

        # 如果指标不存在，先计算
        if not has_smi and len(df) >= 14:
            df = calculate_smi(df)
            has_smi = True

        if not has_stochastic and len(df) >= 14:
            df = calculate_stochastic(df)
            has_stochastic = True

        if not has_sar and len(df) >= 5:
            df = calculate_parabolic_sar(df)
            has_sar = True

        # 获取最新值
        latest = df.iloc[-1]

        # 分析SMI
        if has_smi:
            smi = latest['SMI']
            smi_signal = latest['SMI_Signal']

            # 记录指标值
            result["indicators"]["smi"] = {
                "value": smi,
                "signal": smi_signal,
                "histogram": smi - smi_signal
            }

            # 根据SMI产生信号
            if smi > 40:
                result["signal"] = "BUY"
                result["confidence"] += 0.15
                result["reasons"].append(f"SMI({smi:.2f})位于强劲看多区域(>40)")
            elif smi < -40:
                result["signal"] = "SELL"
                result["confidence"] += 0.15
                result["reasons"].append(f"SMI({smi:.2f})位于强劲看空区域(<-40)")

            # 根据SMI与信号线的交叉产生信号
            if smi > smi_signal and abs(smi) < 60:  # 避免在极端区域产生信号
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "BUY"
                if result["signal"] == "BUY":
                    result["confidence"] += 0.1
                    result["reasons"].append(f"SMI({smi:.2f})上穿信号线({smi_signal:.2f})")
            elif smi < smi_signal and abs(smi) < 60:
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "SELL"
                if result["signal"] == "SELL":
                    result["confidence"] += 0.1
                    result["reasons"].append(f"SMI({smi:.2f})下穿信号线({smi_signal:.2f})")

        # 分析随机指标
        if has_stochastic:
            k = latest['Stochastic_K']
            d = latest['Stochastic_D']

            # 记录指标值
            result["indicators"]["stochastic"] = {
                "k": k,
                "d": d
            }

            # 超买超卖区域
            if k < 20 and d < 20:
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "BUY"
                if result["signal"] == "BUY":
                    result["confidence"] += 0.2
                    result["reasons"].append(f"随机指标处于超卖区域(K:{k:.2f}, D:{d:.2f} < 20)")
            elif k > 80 and d > 80:
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "SELL"
                if result["signal"] == "SELL":
                    result["confidence"] += 0.2
                    result["reasons"].append(f"随机指标处于超买区域(K:{k:.2f}, D:{d:.2f} > 80)")

            # 交叉信号
            cross_up = latest.get('Stochastic_Cross_Up', 0)
            cross_down = latest.get('Stochastic_Cross_Down', 0)

            if cross_up:
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "BUY"
                if result["signal"] == "BUY":
                    result["confidence"] += 0.15
                    result["reasons"].append(f"随机指标K线上穿D线")
            elif cross_down:
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "SELL"
                if result["signal"] == "SELL":
                    result["confidence"] += 0.15
                    result["reasons"].append(f"随机指标K线下穿D线")

        # 分析SAR
        if has_sar:
            sar = latest['SAR']
            price = latest['close']
            trend = latest['SAR_Trend']
            trend_change = latest.get('SAR_Trend_Change', 0)

            # 记录指标值
            result["indicators"]["sar"] = {
                "value": sar,
                "trend": "UP" if trend == 1 else "DOWN",
                "trend_change": trend_change > 0
            }

            # 根据SAR趋势
            if trend == 1:  # 上升趋势
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "BUY"
                if result["signal"] == "BUY":
                    result["confidence"] += 0.2
                    result["reasons"].append(f"SAR指示上升趋势(SAR:{sar:.6f} < 价格:{price:.6f})")
            elif trend == -1:  # 下降趋势
                if result["signal"] == "NEUTRAL":
                    result["signal"] = "SELL"
                if result["signal"] == "SELL":
                    result["confidence"] += 0.2
                    result["reasons"].append(f"SAR指示下降趋势(SAR:{sar:.6f} > 价格:{price:.6f})")

            # 趋势刚刚转向，信号更强
            if trend_change > 0:
                result["confidence"] += 0.1
                if trend == 1:  # 刚转为上升
                    result["reasons"].append(f"SAR刚刚转为上升趋势")
                else:  # 刚转为下降
                    result["reasons"].append(f"SAR刚刚转为下降趋势")

        # 调整最终置信度
        # 限制在0.0-1.0之间
        result["confidence"] = min(1.0, result["confidence"])

        # 信号一致性检查
        if has_smi and has_stochastic and has_sar:
            # 计算各指标的方向
            smi_direction = 1 if latest['SMI'] > 0 else -1
            stoch_direction = 1 if latest['Stochastic_K'] > latest['Stochastic_D'] else -1
            sar_direction = latest['SAR_Trend']

            # 所有指标方向一致
            if smi_direction == stoch_direction == sar_direction:
                result["confidence"] += 0.2
                direction_text = "看多" if smi_direction == 1 else "看空"
                result["reasons"].append(f"所有高级指标方向一致({direction_text})")

        return result
    except Exception as e:
        print_colored(f"❌ 分析高级指标失败: {e}", Colors.ERROR)
        return {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "reasons": [f"分析过程出错: {str(e)}"],
            "error": str(e)
        }


def analyze_vortex_indicator(df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析Vortex指标，生成详细的交易信号和趋势强度评估

    参数:
        df: 包含计算好的Vortex指标的DataFrame

    返回:
        包含分析结果的字典
    """
    try:
        result = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "strength": 0.0,
            "reasons": [],
            "details": {}
        }

        # 检查Vortex指标是否存在
        if not all(col in df.columns for col in ['VI_plus', 'VI_minus']):
            # 如果指标不存在，计算它
            if len(df) >= 14:
                df = calculate_vortex_indicator(df)
            else:
                print_colored("⚠️ 数据不足，无法计算Vortex指标", Colors.WARNING)
                return result

        # 获取最新值
        vi_plus = df['VI_plus'].iloc[-1]
        vi_minus = df['VI_minus'].iloc[-1]
        vi_diff = df['VI_diff'].iloc[-1] if 'VI_diff' in df.columns else vi_plus - vi_minus

        # 检查交叉信号
        vortex_cross_up = df['Vortex_Cross_Up'].iloc[-1] if 'Vortex_Cross_Up' in df.columns else 0
        vortex_cross_down = df['Vortex_Cross_Down'].iloc[-1] if 'Vortex_Cross_Down' in df.columns else 0

        # 记录基础值
        result["details"] = {
            "vi_plus": float(vi_plus),
            "vi_minus": float(vi_minus),
            "vi_diff": float(vi_diff),
            "cross_up": bool(vortex_cross_up),
            "cross_down": bool(vortex_cross_down)
        }

        # 计算趋势强度 - 针对虚拟货币市场优化
        strength = abs(vi_diff) * 10  # 放大差值
        result["strength"] = float(strength)

        # 确定信号
        if vi_plus > vi_minus:
            result["signal"] = "BUY"

            # 计算置信度
            if vortex_cross_up:
                # 刚刚发生上穿 - 较强信号
                result["confidence"] = min(1.0, 0.7 + strength * 0.2)
                result["reasons"].append(f"Vortex VI+上穿VI-，形成新的上升趋势")
            else:
                # 持续上升趋势
                # 根据趋势强度调整置信度
                result["confidence"] = min(1.0, 0.5 + strength * 0.25)
                result["reasons"].append(f"Vortex处于上升趋势，强度: {strength:.2f}")

        elif vi_plus < vi_minus:
            result["signal"] = "SELL"

            # 计算置信度
            if vortex_cross_down:
                # 刚刚发生下穿 - 较强信号
                result["confidence"] = min(1.0, 0.7 + strength * 0.2)
                result["reasons"].append(f"Vortex VI+下穿VI-，形成新的下降趋势")
            else:
                # 持续下降趋势
                result["confidence"] = min(1.0, 0.5 + strength * 0.25)
                result["reasons"].append(f"Vortex处于下降趋势，强度: {strength:.2f}")

        # 虚拟货币市场特殊考量：增加极端趋势识别
        if strength > 2.0:
            result["reasons"].append(f"Vortex显示极强趋势({strength:.2f})，适合顺势交易")
        elif strength < 0.3:
            result["reasons"].append(f"Vortex趋势强度较弱({strength:.2f})，可能处于震荡区间")
            # 弱趋势降低置信度
            result["confidence"] *= 0.7

        # 打印结果
        signal_color = Colors.GREEN if result["signal"] == "BUY" else Colors.RED if result[
                                                                                        "signal"] == "SELL" else Colors.RESET
        print_colored(
            f"Vortex分析结果: {signal_color}{result['signal']}{Colors.RESET}, "
            f"置信度: {result['confidence']:.2f}, 强度: {result['strength']:.2f}",
            Colors.INFO
        )

        for reason in result["reasons"]:
            print_colored(f"• {reason}", Colors.INFO)

        return result

    except Exception as e:
        print_colored(f"❌ 分析Vortex指标失败: {e}", Colors.ERROR)
        return {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "strength": 0.0,
            "reasons": [f"分析出错: {str(e)}"],
            "error": str(e)
        }

def get_advanced_indicator_score(df: pd.DataFrame) -> float:
    """
    基于高级指标计算质量评分

    参数:
        df: 包含高级指标的DataFrame

    返回:
        质量评分 (0-10)
    """
    # 获取综合分析结果
    analysis = analyze_advanced_indicators(df)

    # 基础评分
    base_score = 5.0

    # 根据信号和置信度调整
    signal = analysis["signal"]
    confidence = analysis["confidence"]

    # 如果是看多信号，加分
    if signal == "BUY":
        score_adjustment = 2.0 * confidence
    # 如果是看空信号，减分
    elif signal == "SELL":
        score_adjustment = -2.0 * confidence
    else:
        score_adjustment = 0.0

    # 根据信号一致性调整
    if "所有高级指标方向一致" in " ".join(analysis["reasons"]):
        if signal == "BUY":
            score_adjustment += 1.5
        elif signal == "SELL":
            score_adjustment -= 1.5

    # 计算最终分数
    final_score = base_score + score_adjustment

    # 确保分数在0-10的范围内
    final_score = max(0.0, min(10.0, final_score))

    print_colored(
        f"高级指标质量评分: {final_score:.2f}/10.0, "
        f"信号: {signal}, 置信度: {confidence:.2f}",
        Colors.INFO
    )

    return final_score