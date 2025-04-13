"""
集成模块
整合所有新指标和功能，提供接口用于计算和分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import time

# 导入现有模块
from indicators_module import (
    calculate_optimized_indicators,
    get_smc_trend_and_duration,
    find_swing_points,
    calculate_fibonacci_retracements
)
from logger_utils import Colors, print_colored, log_indicator, log_trend

# 导入新实现的功能模块
from pivot_points_module import (
    calculate_pivot_points,
    analyze_pivot_point_strategy,
    get_pivot_points_quality_score
)

from advanced_indicators import (
    calculate_smi,
    calculate_stochastic,
    calculate_parabolic_sar,
    analyze_advanced_indicators,
    get_advanced_indicator_score
)

from smc_enhanced_prediction import (
    enhanced_smc_prediction,
    multi_timeframe_smc_prediction,
    calculate_optimal_holding_time
)

from risk_management import (
    calculate_leveraged_stop_loss,
    calculate_dynamic_take_profit,
    advanced_smc_stop_loss,
    calculate_trailing_stop_params,
    calculate_position_size,
    adaptive_risk_management
)

from entry_timing_module import (
    calculate_entry_timing,
    detect_breakout_conditions,
    estimate_entry_execution_price
)


def calculate_enhanced_indicators(df: pd.DataFrame,
                                  calculate_all: bool = True,
                                  include_pivot: bool = True,
                                  include_advanced: bool = True,
                                  pivot_method: str = 'classic') -> pd.DataFrame:
    """
    增强版指标计算函数，整合所有指标计算

    参数:
        df: 价格数据DataFrame
        calculate_all: 是否计算所有基础指标
        include_pivot: 是否包含支点分析
        include_advanced: 是否包含高级指标
        pivot_method: 支点计算方法 ('classic', 'woodie', 'camarilla')

    返回:
        df: 添加了所有指标的DataFrame
    """
    start_time = time.time()
    print_colored("🔄 开始计算增强版指标...", Colors.BLUE + Colors.BOLD)

    try:
        # 1. 首先计算基础优化指标
        if calculate_all:
            df = calculate_optimized_indicators(df)

            if df is None or df.empty:
                print_colored("❌ 基础指标计算失败", Colors.ERROR)
                return pd.DataFrame()

        # 2. 计算支点指标
        if include_pivot:
            df = calculate_pivot_points(df, method=pivot_method)
            print_colored(f"✅ {pivot_method.capitalize()}支点计算完成", Colors.GREEN)

        # 3. 计算高级指标
        if include_advanced:
            # 随机动量指数 (SMI)
            if len(df) >= 14:
                df = calculate_smi(df)
                print_colored("✅ SMI计算完成", Colors.GREEN)

            # 随机指标 (Stochastic Oscillator)
            if len(df) >= 14:
                df = calculate_stochastic(df)
                print_colored("✅ 随机指标计算完成", Colors.GREEN)

            # 抛物线转向指标 (Parabolic SAR)
            if len(df) >= 5:
                df = calculate_parabolic_sar(df)
                print_colored("✅ 抛物线SAR计算完成", Colors.GREEN)

        execution_time = time.time() - start_time
        print_colored(f"✅ 所有指标计算完成，耗时: {execution_time:.2f}秒", Colors.GREEN + Colors.BOLD)

        return df
    except Exception as e:
        print_colored(f"❌ 增强版指标计算失败: {e}", Colors.ERROR)
        # 尝试恢复部分功能
        if df is not None and not df.empty and 'close' in df.columns:
            return df
        return pd.DataFrame()


def comprehensive_market_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    全面市场分析，整合所有指标和分析结果

    参数:
        df: 包含所有指标的DataFrame

    返回:
        包含完整分析结果的字典
    """
    print_colored("🔍 开始全面市场分析...", Colors.BLUE + Colors.BOLD)

    try:
        result = {
            "timestamp": time.time(),
            "indicators": {},
            "signals": {},
            "predictions": {},
            "quality_scores": {},
            "overall": {}
        }

        # 确保数据有效
        if df is None or df.empty or len(df) < 20:
            print_colored("❌ 数据不足，无法进行分析", Colors.ERROR)
            return {"error": "insufficient_data"}

        # 获取当前价格
        current_price = df['close'].iloc[-1]
        result["current_price"] = current_price

        # 1. SMC趋势分析
        trend, duration, trend_info = get_smc_trend_and_duration(df)
        result["trend"] = {
            "direction": trend,
            "duration": duration,
            "confidence": trend_info["confidence"],
            "reason": trend_info.get("reason", "未知")
        }

        # 2. 支点分析
        pivot_analysis = analyze_pivot_point_strategy(df)
        result["signals"]["pivot"] = pivot_analysis

        # 3. 高级指标分析
        advanced_analysis = analyze_advanced_indicators(df)
        result["signals"]["advanced"] = advanced_analysis

        # 4. 多时间框架预测
        predictions = multi_timeframe_smc_prediction(df)
        if "error" not in predictions:
            result["predictions"] = predictions

        # 5. 质量评分计算
        # 5.1 支点质量评分
        pivot_score = get_pivot_points_quality_score(df)
        result["quality_scores"]["pivot"] = pivot_score

        # 5.2 高级指标质量评分
        advanced_score = get_advanced_indicator_score(df)
        result["quality_scores"]["advanced"] = advanced_score

        # 5.3 计算综合质量评分
        # 不同指标分数的权重
        weights = {
            "pivot": 0.3,  # 支点分析权重
            "advanced": 0.3,  # 高级指标权重
            "trend": 0.4  # 趋势分析权重
        }

        # 趋势质量转换为分数
        trend_score = 0
        if trend == "UP":
            trend_score = 7.0  # 上升趋势基础分
            if trend_info["confidence"] == "高":
                trend_score = 10.0
            elif trend_info["confidence"] == "中高":
                trend_score = 9.0
            elif trend_info["confidence"] == "中":
                trend_score = 8.0
        elif trend == "DOWN":
            trend_score = 3.0  # 下降趋势基础分
            if trend_info["confidence"] == "高":
                trend_score = 0.0
            elif trend_info["confidence"] == "中高":
                trend_score = 1.0
            elif trend_info["confidence"] == "中":
                trend_score = 2.0
        else:  # NEUTRAL
            trend_score = 5.0  # 中性趋势

        # 计算综合质量评分
        composite_score = (
                weights["pivot"] * pivot_score["score"] +
                weights["advanced"] * advanced_score +
                weights["trend"] * trend_score
        )

        result["quality_scores"]["trend"] = trend_score
        result["quality_scores"]["composite"] = composite_score

        # 6. 生成交易信号和建议
        # 6.1 整合各指标信号
        signals = []
        confidences = []

        # 添加支点信号
        pivot_signal = pivot_analysis["signal"]
        pivot_confidence = pivot_analysis["confidence"]
        if pivot_signal != "NEUTRAL":
            signals.append(pivot_signal)
            confidences.append(pivot_confidence)

        # 添加高级指标信号
        adv_signal = advanced_analysis["signal"]
        adv_confidence = advanced_analysis["confidence"]
        if adv_signal != "NEUTRAL":
            signals.append(adv_signal)
            confidences.append(adv_confidence)

        # 添加趋势信号
        if trend == "UP":
            signals.append("BUY")
            trend_confidence = 0.7 if trend_info["confidence"] == "高" else 0.5 if trend_info[
                                                                                       "confidence"] == "中高" else 0.3
            confidences.append(trend_confidence)
        elif trend == "DOWN":
            signals.append("SELL")
            trend_confidence = 0.7 if trend_info["confidence"] == "高" else 0.5 if trend_info[
                                                                                       "confidence"] == "中高" else 0.3
            confidences.append(trend_confidence)

        # 6.2 判断最终信号
        if not signals:
            final_signal = "NEUTRAL"
            final_confidence = 0.0
            signal_reason = "无明确信号"
        else:
            # 计算买卖信号的总置信度
            buy_confidence = sum(confidences[i] for i in range(len(signals)) if signals[i] == "BUY")
            sell_confidence = sum(confidences[i] for i in range(len(signals)) if signals[i] == "SELL")

            if buy_confidence > sell_confidence:
                final_signal = "BUY"
                final_confidence = buy_confidence / len([s for s in signals if s == "BUY"])
                signal_reason = "买入信号强度大于卖出信号"
            elif sell_confidence > buy_confidence:
                final_signal = "SELL"
                final_confidence = sell_confidence / len([s for s in signals if s == "SELL"])
                signal_reason = "卖出信号强度大于买入信号"
            else:
                # 买卖信号强度相等或无信号
                final_signal = "NEUTRAL"
                final_confidence = 0.0
                signal_reason = "买卖信号强度相等"

        # 6.3 基于质量评分调整信号置信度
        signal_quality_factor = composite_score / 10.0  # 将质量评分转换为0-1的因子

        # 只有在方向一致时才提升置信度
        if (final_signal == "BUY" and composite_score >= 6.0) or (final_signal == "SELL" and composite_score <= 4.0):
            final_confidence = min(1.0, final_confidence * (1 + signal_quality_factor * 0.5))
        elif (final_signal == "BUY" and composite_score < 5.0) or (final_signal == "SELL" and composite_score > 5.0):
            # 信号与质量评分不一致，降低置信度
            final_confidence = final_confidence * 0.7
            signal_reason += "，但与质量评分不一致"

        # 记录最终结果
        result["overall"] = {
            "signal": final_signal,
            "confidence": final_confidence,
            "reason": signal_reason,
            "quality_score": composite_score,
            "timestamp": time.time()
        }

        # 打印分析总结
        print_colored("📊 市场分析总结:", Colors.BLUE + Colors.BOLD)
        print_colored(f"综合质量评分: {composite_score:.2f}/10", Colors.INFO)

        signal_color = (
            Colors.GREEN if final_signal == "BUY" else
            Colors.RED if final_signal == "SELL" else
            Colors.GRAY
        )

        print_colored(
            f"最终信号: {signal_color}{final_signal}{Colors.RESET}, "
            f"置信度: {final_confidence:.2f}",
            Colors.INFO
        )
        print_colored(f"信号原因: {signal_reason}", Colors.INFO)

        return result
    except Exception as e:
        print_colored(f"❌ 市场分析失败: {e}", Colors.ERROR)
        return {"error": str(e)}


def generate_trade_recommendation(df: pd.DataFrame, account_balance: float,
                                  leverage: int = 1) -> Dict[str, Any]:
    """
    生成完整交易建议，包括入场、止损、止盈、仓位大小等

    参数:
        df: 价格数据DataFrame (需已计算所有指标)
        account_balance: 账户余额
        leverage: 杠杆倍数

    返回:
        包含完整交易建议的字典
    """
    print_colored("🎯 生成交易建议...", Colors.BLUE + Colors.BOLD)

    try:
        # 1. 进行全面市场分析
        analysis = comprehensive_market_analysis(df)

        # 检查是否有错误
        if "error" in analysis:
            return {"error": analysis["error"], "recommendation": "AVOID"}

        # 2. 确定交易方向
        signal = analysis["overall"]["signal"]
        confidence = analysis["overall"]["confidence"]
        quality_score = analysis["overall"]["quality_score"]
        current_price = analysis["current_price"]

        # 如果没有明确信号或质量评分过低，建议避免交易
        if signal == "NEUTRAL" or confidence < 0.3:
            return {
                "recommendation": "AVOID",
                "reason": "无明确交易信号或置信度过低",
                "analysis": analysis
            }

        if signal == "BUY" and quality_score < 5.0:
            return {
                "recommendation": "AVOID",
                "reason": "买入信号质量评分过低",
                "analysis": analysis
            }

        if signal == "SELL" and quality_score > 5.0:
            return {
                "recommendation": "AVOID",
                "reason": "卖出信号质量评分过高",
                "analysis": analysis
            }

        # 3. 计算入场时机
        entry_timing = calculate_entry_timing(df, signal, quality_score, current_price)

        # 4. 检测突破条件
        breakout = detect_breakout_conditions(df)

        # 5. 进行风险管理分析
        risk_analysis = adaptive_risk_management(df, account_balance, quality_score, signal, leverage)

        # 如果风险管理建议避免交易
        if risk_analysis.get("recommendation") in ["AVOID", "REDUCE_LEVERAGE"]:
            return {
                "recommendation": risk_analysis["recommendation"],
                "reason": risk_analysis["recommendation_reason"],
                "analysis": analysis,
                "risk": risk_analysis,
                "entry_timing": entry_timing
            }

        # 6. 计算最佳持仓时间
        optimal_holding_time = calculate_optimal_holding_time(df, analysis["trend"])

        # 7. 确定入场价格和方式
        # 优先考虑入场时机建议的价格
        entry_price = entry_timing["expected_entry_price"]
        entry_type = entry_timing["entry_type"]
        should_wait = entry_timing["should_wait"]

        # 如果有突破且方向与信号一致，可能需要调整入场价
        if breakout["has_breakout"] and ((breakout["direction"] == "UP" and signal == "BUY") or
                                         (breakout["direction"] == "DOWN" and signal == "SELL")):
            # 如果是强突破，考虑立即入场
            if breakout["strength"] > 1.0:
                entry_timing["immediate_entry"] = True
                entry_timing["should_wait"] = False
                entry_timing["entry_type"] = "MARKET"
                should_wait = False
                entry_type = "MARKET"

                # 调整入场价考虑市场冲击
                entry_price = estimate_entry_execution_price(
                    current_price, signal, "MARKET", market_impact=0.001)

                # 添加突破理由
                entry_timing["entry_conditions"].append(
                    f"检测到{breakout['direction']}方向突破: {breakout['description']}")

        # 8. 使用多时间框架预测优化入场和出场
        if "predictions" in analysis and "optimal_trading_zone" in analysis["predictions"]:
            trading_zone = analysis["predictions"]["optimal_trading_zone"]

            # 如果多时间框架预测与信号一致，使用其建议的止损止盈
            if trading_zone.get("recommendation") == signal:
                stop_loss = trading_zone.get("stop_loss", risk_analysis["stop_loss"])
                take_profit = trading_zone.get("take_profit", risk_analysis["take_profit"])
            else:
                # 否则使用风险分析结果
                stop_loss = risk_analysis["stop_loss"]
                take_profit = risk_analysis["take_profit"]
        else:
            # 使用风险分析结果
            stop_loss = risk_analysis["stop_loss"]
            take_profit = risk_analysis["take_profit"]

        # 9. 计算实际入场价的仓位大小
        # 如果预期入场价不是当前价，需要调整仓位计算
        if abs(entry_price - current_price) / current_price > 0.005:
            # 重新计算仓位大小
            position_result = calculate_position_size(
                account_balance,
                entry_price,
                stop_loss,
                risk_analysis["max_risk_percent"],
                leverage
            )
            position_size = position_result["position_size"]
            position_value = position_result["position_value"]
        else:
            position_size = risk_analysis["position_size"]
            position_value = risk_analysis["position_value"]

        # 10. 整合所有信息生成最终建议
        recommendation = {
            "recommendation": "EXECUTE" if not should_wait or entry_timing["immediate_entry"] else "WAIT",
            "side": signal,
            "confidence": confidence,
            "quality_score": quality_score,
            "entry_price": entry_price,
            "entry_type": entry_type,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "position_value": position_value,
            "leverage": leverage,
            "risk_percent": risk_analysis["actual_risk_percent"],
            "risk_level": risk_analysis["risk_level"],
            "optimal_holding_time": optimal_holding_time,
            "trailing_stop": risk_analysis["trailing_stop"],
            "risk_reward_ratio": risk_analysis["risk_reward_ratio"],
            "entry_timing": entry_timing,
            "breakout": breakout if breakout["has_breakout"] else None,
            "analysis": analysis,
            "timestamp": time.time()
        }

        # 打印建议摘要
        print_colored("💼 交易建议摘要:", Colors.BLUE + Colors.BOLD)

        signal_color = Colors.GREEN if signal == "BUY" else Colors.RED if signal == "SELL" else Colors.GRAY
        action_text = "立即执行" if recommendation[
                                        "recommendation"] == "EXECUTE" else f"等待入场 ({entry_timing['expected_entry_minutes']}分钟)"

        print_colored(f"建议: {action_text} {signal_color}{signal}{Colors.RESET}", Colors.INFO)
        print_colored(f"当前价格: {current_price:.6f}", Colors.INFO)
        print_colored(f"入场价格: {entry_price:.6f} ({entry_type}单)", Colors.INFO)

        print_colored("入场条件:", Colors.BLUE)
        for i, condition in enumerate(entry_timing["entry_conditions"], 1):
            print_colored(f"  {i}. {condition}", Colors.INFO)

        print_colored(f"预计入场时间: {entry_timing['expected_entry_time']}", Colors.INFO)
        print_colored(f"止损价格: {stop_loss:.6f}", Colors.INFO)
        print_colored(f"止盈价格: {take_profit:.6f}", Colors.INFO)
        print_colored(f"仓位规模: {position_size:.6f} 单位", Colors.INFO)
        print_colored(f"仓位价值: {position_value:.2f}", Colors.INFO)
        print_colored(f"风险: {risk_analysis['actual_risk_percent']:.2f}%", Colors.INFO)
        print_colored(f"风险回报比: {risk_analysis['risk_reward_ratio']:.2f}", Colors.INFO)
        print_colored(f"最佳持仓时间: {optimal_holding_time}分钟", Colors.INFO)

        if breakout["has_breakout"]:
            b_color = Colors.GREEN if breakout["direction"] == "UP" else Colors.RED
            print_colored(
                f"突破情况: {b_color}{breakout['description']}{Colors.RESET} (强度: {breakout['strength']:.2f})",
                Colors.INFO)

        return recommendation
    except Exception as e:
        print_colored(f"❌ 生成交易建议失败: {e}", Colors.ERROR)
        return {"error": str(e), "recommendation": "AVOID"}