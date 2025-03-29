"""
增强型SMC预测模块
提供多时间框架(Smart Money Concept)预测功能，结合关键价格区域和市场结构分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from logger_utils import Colors, print_colored
from indicators_module import find_swing_points, calculate_fibonacci_retracements, get_smc_trend_and_duration


def enhanced_smc_prediction(df: pd.DataFrame, horizon: str = 'medium', config: Optional[Dict[str, Any]] = None) -> Dict[
    str, Any]:
    """
    SMC增强预测方法，提供多时间框架市场预测

    参数:
        df: 价格数据DataFrame
        horizon: 预测时间范围 ('short', 'medium', 'long')
        config: 配置参数

    返回:
        包含预测结果的字典
    """
    if df is None or len(df) < 20:
        print_colored("⚠️ 数据不足，无法进行SMC预测分析", Colors.WARNING)
        return {"error": "insufficient_data"}

    # 默认配置
    default_config = {
        "horizons": {
            "short": 30,  # 30分钟
            "medium": 240,  # 4小时
            "long": 1440  # 24小时
        }
    }

    # 合并配置
    if config is None:
        config = {}

    effective_config = {**default_config, **config}

    # 获取预测窗口
    window_length = effective_config["horizons"].get(horizon, 240)

    print_colored(f"执行增强型SMC预测分析: {horizon}期 ({window_length}分钟)", Colors.INFO)

    try:
        # 多维度分析
        swing_highs, swing_lows = find_swing_points(df)
        fib_levels = calculate_fibonacci_retracements(df)

        # 趋势分析
        trend, duration, trend_info = get_smc_trend_and_duration(df, config, None)

        # 计算关键价格区域
        current_price = df['close'].iloc[-1]

        # 获取近期的支撑位和阻力位
        recent_swing_highs = sorted([h for h in swing_highs if h > current_price])
        recent_swing_lows = sorted([l for l in swing_lows if l < current_price], reverse=True)

        # 获取最近的支撑位和阻力位
        support = recent_swing_lows[0] if recent_swing_lows else df['low'].min()
        resistance = recent_swing_highs[0] if recent_swing_highs else df['high'].max()

        # 价格区间宽度
        price_range = resistance - support

        # 设置预测时间窗口基于K线类型
        candle_minutes = 15  # 假设使用15分钟K线
        prediction_candles = window_length / candle_minutes

        # 计算短期趋势斜率
        if len(df) > 5:
            recent_prices = df['close'].tail(5).values
            slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            slope_pct = slope / current_price * 100  # 每K线价格变化百分比
        else:
            slope = 0
            slope_pct = 0

        # 预测价格变动
        predicted_change_pct = slope_pct * prediction_candles

        # 根据趋势信息强化预测
        confidence = 1.0
        if trend_info["confidence"] == "高":
            confidence = 1.5
        elif trend_info["confidence"] == "中高":
            confidence = 1.2
        elif trend_info["confidence"] == "中":
            confidence = 1.0
        elif trend_info["confidence"] == "低":
            confidence = 0.7

        # 调整预测变化
        adjusted_change_pct = predicted_change_pct * confidence

        # 计算预测价格
        base_prediction = current_price * (1 + adjusted_change_pct / 100)

        # 确保预测价格在合理范围内
        if trend == "UP":
            min_price = current_price
            max_price = min(resistance, current_price * 1.2)  # 不超过阻力位或当前价格的20%
        elif trend == "DOWN":
            max_price = current_price
            min_price = max(support, current_price * 0.8)  # 不低于支撑位或当前价格的-20%
        else:  # NEUTRAL
            min_price = max(support, current_price * 0.95)
            max_price = min(resistance, current_price * 1.05)

        # 调整最终预测价格，确保在合理范围内
        final_prediction = max(min_price, min(max_price, base_prediction))

        # 基于斐波那契水平的预测区域
        prediction_zones = {}
        if fib_levels and len(fib_levels) >= 3:
            # 根据趋势方向过滤斐波那契水平
            if trend == "UP":
                relevant_fibs = [level for level in fib_levels if level > current_price]
                if relevant_fibs:
                    prediction_zones["fib_resistance_1"] = min(relevant_fibs)
                    if len(relevant_fibs) > 1:
                        prediction_zones["fib_resistance_2"] = sorted(relevant_fibs)[1]
            elif trend == "DOWN":
                relevant_fibs = [level for level in fib_levels if level < current_price]
                if relevant_fibs:
                    prediction_zones["fib_support_1"] = max(relevant_fibs)
                    if len(relevant_fibs) > 1:
                        prediction_zones["fib_support_2"] = sorted(relevant_fibs, reverse=True)[1]

        # 计算交易区间
        low_risk_zone = {}
        high_probability_zone = {}
        high_risk_zone = {}

        if trend == "UP":
            # 上升趋势
            low_risk_zone = {
                "entry": support,
                "stop_loss": support * 0.98,
                "take_profit": current_price
            }

            high_probability_zone = {
                "entry": current_price,
                "stop_loss": max(support, current_price * 0.95),
                "take_profit": min(resistance, current_price * 1.1)
            }

            high_risk_zone = {
                "entry": current_price * 1.02,
                "stop_loss": current_price * 0.97,
                "take_profit": resistance
            }
        elif trend == "DOWN":
            # 下降趋势
            low_risk_zone = {
                "entry": resistance,
                "stop_loss": resistance * 1.02,
                "take_profit": current_price
            }

            high_probability_zone = {
                "entry": current_price,
                "stop_loss": min(resistance, current_price * 1.05),
                "take_profit": max(support, current_price * 0.9)
            }

            high_risk_zone = {
                "entry": current_price * 0.98,
                "stop_loss": current_price * 1.03,
                "take_profit": support
            }
        else:
            # 中性趋势 - 考虑区间交易
            mid_point = (support + resistance) / 2
            range_half = (resistance - support) / 2

            low_risk_zone = {
                "entry": current_price,
                "stop_loss": current_price * (0.97 if current_price > mid_point else 1.03),
                "take_profit": current_price * (1.05 if current_price > mid_point else 0.95)
            }

            high_probability_zone = {
                "entry": support if current_price < mid_point else resistance,
                "stop_loss": support * 0.97 if current_price < mid_point else resistance * 1.03,
                "take_profit": mid_point
            }

            high_risk_zone = {
                "entry": mid_point,
                "stop_loss": mid_point * (0.97 if current_price < mid_point else 1.03),
                "take_profit": resistance if current_price < mid_point else support
            }

        # 构建结果
        result = {
            "trend": trend,
            "confidence": trend_info["confidence"],
            "current_price": current_price,
            "predicted_price": final_prediction,
            "predicted_change_pct": (final_prediction - current_price) / current_price * 100,
            "support": support,
            "resistance": resistance,
            "time_horizon": {
                "name": horizon,
                "minutes": window_length
            },
            "prediction_zones": prediction_zones,
            "trading_zones": {
                "low_risk": low_risk_zone,
                "high_probability": high_probability_zone,
                "high_risk": high_risk_zone
            }
        }

        # 打印预测结果
        print_colored(f"SMC预测结果 ({horizon}期):", Colors.BLUE + Colors.BOLD)
        print_colored(
            f"趋势: {Colors.GREEN if trend == 'UP' else Colors.RED if trend == 'DOWN' else Colors.GRAY}{trend}{Colors.RESET}, "
            f"置信度: {trend_info['confidence']}",
            Colors.INFO
        )
        print_colored(
            f"当前价格: {current_price:.6f}, 预测价格: {final_prediction:.6f} "
            f"({(final_prediction - current_price) / current_price * 100:+.2f}%)",
            Colors.INFO
        )
        print_colored(f"支撑位: {support:.6f}, 阻力位: {resistance:.6f}", Colors.INFO)

        return result
    except Exception as e:
        print_colored(f"❌ SMC预测分析失败: {e}", Colors.ERROR)
        return {"error": str(e)}


def multi_timeframe_smc_prediction(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    执行多时间框架SMC预测分析，提供短、中、长期的市场预测结果

    参数:
        df: 价格数据DataFrame
        config: 配置参数

    返回:
        包含多时间框架预测结果的字典
    """
    if df is None or len(df) < 20:
        print_colored("⚠️ 数据不足，无法进行多时间框架SMC预测分析", Colors.WARNING)
        return {"error": "insufficient_data"}

    print_colored("执行多时间框架SMC预测分析", Colors.BLUE + Colors.BOLD)

    try:
        # 短期预测 (15-30分钟)
        short_term = enhanced_smc_prediction(df, "short", config)

        # 中期预测 (4小时)
        medium_term = enhanced_smc_prediction(df, "medium", config)

        # 长期预测 (24小时)
        long_term = enhanced_smc_prediction(df, "long", config)

        # 计算综合趋势和置信度
        trends = {
            result["trend"]: 0
            for result in [short_term, medium_term, long_term]
            if "trend" in result
        }

        for result in [short_term, medium_term, long_term]:
            if "trend" in result:
                weight = 1
                if result["time_horizon"]["name"] == "short":
                    weight = 1
                elif result["time_horizon"]["name"] == "medium":
                    weight = 2
                elif result["time_horizon"]["name"] == "long":
                    weight = 3

                trends[result["trend"]] += weight

        # 确定主导趋势
        dominant_trend = max(trends.items(), key=lambda x: x[1])[0] if trends else "NEUTRAL"

        # 趋势一致性
        trend_consistency = 1.0
        if len(set(result["trend"] for result in [short_term, medium_term, long_term] if "trend" in result)) == 1:
            # 所有时间框架趋势一致
            trend_consistency = 2.0
            trend_consistency_text = "高度一致"
        elif dominant_trend != "NEUTRAL" and all(
                result.get("trend") != "NEUTRAL" for result in [short_term, medium_term, long_term] if
                "trend" in result):
            # 没有中性趋势，但有不同方向
            trend_consistency = 0.5
            trend_consistency_text = "趋势冲突"
        else:
            trend_consistency_text = "部分一致"

        # 计算最优入场和出场区域
        common_trading_zones = {}

        # 根据主导趋势选择交易区域
        if dominant_trend == "UP":
            zone_type = "high_probability" if trend_consistency >= 1.0 else "low_risk"
            optimal_zones = medium_term.get("trading_zones", {}).get(zone_type, {})

            common_trading_zones = {
                "entry": optimal_zones.get("entry"),
                "stop_loss": optimal_zones.get("stop_loss"),
                "take_profit": optimal_zones.get("take_profit"),
                "recommendation": "BUY",
                "confidence": trend_consistency
            }
        elif dominant_trend == "DOWN":
            zone_type = "high_probability" if trend_consistency >= 1.0 else "low_risk"
            optimal_zones = medium_term.get("trading_zones", {}).get(zone_type, {})

            common_trading_zones = {
                "entry": optimal_zones.get("entry"),
                "stop_loss": optimal_zones.get("stop_loss"),
                "take_profit": optimal_zones.get("take_profit"),
                "recommendation": "SELL",
                "confidence": trend_consistency
            }
        else:
            common_trading_zones = {
                "recommendation": "NEUTRAL",
                "confidence": 0.5,
                "note": "趋势不明确，建议观望"
            }

        # 构建综合结果
        result = {
            "short_term": short_term,
            "medium_term": medium_term,
            "long_term": long_term,
            "dominant_trend": dominant_trend,
            "trend_consistency": trend_consistency,
            "trend_consistency_text": trend_consistency_text,
            "optimal_trading_zone": common_trading_zones
        }

        # 打印综合分析结果
        print_colored("多时间框架分析结果:", Colors.BLUE + Colors.BOLD)
        print_colored(
            f"主导趋势: {Colors.GREEN if dominant_trend == 'UP' else Colors.RED if dominant_trend == 'DOWN' else Colors.GRAY}{dominant_trend}{Colors.RESET}, "
            f"趋势一致性: {trend_consistency_text} ({trend_consistency:.1f})",
            Colors.INFO
        )

        if "recommendation" in common_trading_zones:
            rec = common_trading_zones["recommendation"]
            confidence = common_trading_zones.get("confidence", 0)
            rec_color = Colors.GREEN if rec == "BUY" else Colors.RED if rec == "SELL" else Colors.GRAY

            print_colored(
                f"交易建议: {rec_color}{rec}{Colors.RESET}, 置信度: {confidence:.1f}",
                Colors.INFO
            )

            if "entry" in common_trading_zones:
                print_colored(
                    f"建议入场价: {common_trading_zones['entry']:.6f}, "
                    f"止损价: {common_trading_zones['stop_loss']:.6f}, "
                    f"止盈价: {common_trading_zones['take_profit']:.6f}",
                    Colors.INFO
                )

        return result
    except Exception as e:
        print_colored(f"❌ 多时间框架SMC预测分析失败: {e}", Colors.ERROR)
        return {"error": str(e)}


def calculate_optimal_holding_time(df: pd.DataFrame, trend_info: Dict[str, Any]) -> int:
    """
    基于市场结构优化持仓时间

    参数:
        df: 价格数据DataFrame
        trend_info: 趋势信息字典

    返回:
        最佳持仓时间（分钟）
    """
    # 趋势持续时间
    trend_duration = trend_info.get('duration', 0)

    # 趋势置信度
    confidence = trend_info.get('confidence', '低')

    # 支撑/阻力位分析
    swing_highs, swing_lows = find_swing_points(df)

    # 持仓时间映射
    holding_time_map = {
        '高': min(max(trend_duration, 240), 720),  # 4-12小时
        '中高': min(max(trend_duration, 180), 480),  # 3-8小时
        '中': min(max(trend_duration, 120), 360),  # 2-6小时
        '低': min(max(trend_duration, 60), 240)  # 1-4小时
    }

    # 选择持仓时间
    optimal_time = holding_time_map.get(confidence, 240)

    # 额外条件：支撑/阻力位数量增加持仓时间
    time_multiplier = 1 + (len(swing_highs) + len(swing_lows)) * 0.1
    optimal_time = min(optimal_time * time_multiplier, 720)  # 最长12小时

    print_colored("SMC持仓时间优化:", Colors.BLUE)
    print_colored(f"趋势持续时间: {trend_duration}分钟", Colors.INFO)
    print_colored(f"趋势置信度: {confidence}", Colors.INFO)
    print_colored(f"支撑/阻力位数量: {len(swing_highs) + len(swing_lows)}", Colors.INFO)
    print_colored(f"最佳持仓时间: {optimal_time:.0f}分钟", Colors.INFO)

    return int(optimal_time)