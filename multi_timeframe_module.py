"""
多时间框架协调模块
提供不同时间框架数据的获取、分析和一致性评估功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from logger_utils import Colors, print_colored
from indicators_module import calculate_optimized_indicators, get_smc_trend_and_duration


class MultiTimeframeCoordinator:
    """多时间框架协调类，用于在不同时间框架上进行分析并协调决策"""

    def __init__(self, client, logger=None):
        """初始化多时间框架协调器

        参数:
            client: Binance客户端
            logger: 日志对象
        """
        self.client = client
        self.logger = logger
        self.timeframes = {
            "1m": {"interval": "1m", "weight": 0.5, "data": {}, "last_update": {}},
            "5m": {"interval": "5m", "weight": 0.7, "data": {}, "last_update": {}},
            "15m": {"interval": "15m", "weight": 1.0, "data": {}, "last_update": {}},
            "1h": {"interval": "1h", "weight": 1.5, "data": {}, "last_update": {}},
            "4h": {"interval": "4h", "weight": 2.0, "data": {}, "last_update": {}}
        }
        self.update_interval = {
            "1m": 60,  # 1分钟K线每1分钟更新一次
            "5m": 300,  # 5分钟K线每5分钟更新一次
            "15m": 600,  # 15分钟K线每10分钟更新一次
            "1h": 1800,  # 1小时K线每30分钟更新一次
            "4h": 3600  # 4小时K线每60分钟更新一次
        }
        self.coherence_cache = {}  # 缓存一致性分析结果

        print_colored("🔄 多时间框架协调器初始化完成", Colors.GREEN)

    def fetch_all_timeframes(self, symbol: str, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """获取指定交易对的所有时间框架数据

        参数:
            symbol: 交易对
            force_refresh: 是否强制刷新缓存

        返回:
            各时间框架的DataFrame字典
        """
        result = {}
        current_time = time.time()

        print_colored(f"🔍 获取{symbol}的多时间框架数据{'(强制刷新)' if force_refresh else ''}", Colors.BLUE)

        for tf_name, tf_info in self.timeframes.items():
            # 检查是否需要更新数据
            last_update = tf_info["last_update"].get(symbol, 0)
            interval_seconds = self.update_interval[tf_name]

            if force_refresh or (current_time - last_update) > interval_seconds or symbol not in tf_info["data"]:
                try:
                    # 根据时间框架调整获取的K线数量
                    limit = 100
                    if tf_name in ["1h", "4h"]:
                        limit = 200  # 长周期获取更多数据

                    # 获取K线数据
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=tf_info["interval"],
                        limit=limit
                    )

                    # 处理数据
                    df = pd.DataFrame(klines, columns=[
                        'time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'trades',
                        'taker_base_vol', 'taker_quote_vol', 'ignore'
                    ])

                    # 转换数据类型
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

                    # 转换时间
                    df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')

                    # 计算指标
                    df = calculate_optimized_indicators(df)

                    # 缓存数据
                    tf_info["data"][symbol] = df
                    tf_info["last_update"][symbol] = current_time

                    print_colored(f"✅ {tf_name}时间框架数据获取成功: {len(df)}行", Colors.GREEN)
                except Exception as e:
                    print_colored(f"❌ 获取{symbol} {tf_name}数据失败: {e}", Colors.ERROR)
                    if symbol in tf_info["data"]:
                        print_colored(f"使用缓存的{tf_name}数据: {len(tf_info['data'][symbol])}行", Colors.YELLOW)
                    else:
                        tf_info["data"][symbol] = pd.DataFrame()  # 放入空DataFrame避免后续错误
            else:
                print_colored(f"使用缓存的{tf_name}数据: {len(tf_info['data'][symbol])}行", Colors.CYAN)

            # 添加到结果
            result[tf_name] = tf_info["data"][symbol]

        return result

    def analyze_timeframe_trends(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[
        str, Dict[str, Any]]:
        """分析各时间框架的趋势

        参数:
            symbol: 交易对
            timeframe_data: 各时间框架的DataFrame字典

        返回:
            各时间框架的趋势分析结果
        """
        trends = {}

        print_colored(f"📊 分析{symbol}在各时间框架上的趋势", Colors.BLUE)

        for tf_name, df in timeframe_data.items():
            if df.empty:
                print_colored(f"⚠️ {tf_name}数据为空，无法分析趋势", Colors.WARNING)
                trends[tf_name] = {
                    "trend": "UNKNOWN",
                    "duration": 0,
                    "confidence": "无",
                    "valid": False
                }
                continue

            try:
                # 计算趋势
                trend, duration, trend_info = get_smc_trend_and_duration(df)

                # 转换持续时间到该时间框架的周期数
                if tf_name == "1m":
                    periods = duration  # 1分钟就是周期数
                elif tf_name == "5m":
                    periods = duration / 5
                elif tf_name == "15m":
                    periods = duration / 15
                elif tf_name == "1h":
                    periods = duration / 60
                elif tf_name == "4h":
                    periods = duration / 240

                # 趋势颜色
                trend_color = Colors.GREEN if trend == "UP" else Colors.RED if trend == "DOWN" else Colors.GRAY

                print_colored(
                    f"{tf_name}: 趋势 {trend_color}{trend}{Colors.RESET}, "
                    f"持续 {duration}分钟 ({periods:.1f}个周期), "
                    f"置信度: {trend_info['confidence']}",
                    Colors.INFO
                )

                trends[tf_name] = {
                    "trend": trend,
                    "duration": duration,
                    "periods": periods,
                    "confidence": trend_info["confidence"],
                    "reason": trend_info.get("reason", ""),
                    "valid": True,
                    "indicators": trend_info.get("indicators", {})
                }
            except Exception as e:
                print_colored(f"❌ 分析{symbol} {tf_name}趋势失败: {e}", Colors.ERROR)
                trends[tf_name] = {
                    "trend": "UNKNOWN",
                    "duration": 0,
                    "confidence": "无",
                    "valid": False,
                    "error": str(e)
                }

        return trends

    def calculate_timeframe_coherence(self, symbol: str, trend_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算时间框架一致性

        参数:
            symbol: 交易对
            trend_analysis: 趋势分析结果

        返回:
            一致性分析结果
        """
        # 初始化结果
        result = {
            "coherence_score": 0.0,
            "trend_agreement": 0.0,
            "dominant_timeframe": None,
            "dominant_trend": None,
            "trend_conflicts": [],
            "agreement_level": "无",
            "recommendation": "NEUTRAL"
        }

        # 收集有效的趋势
        valid_trends = {}
        trend_counts = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        weighted_scores = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        confidence_weights = {"高": 1.0, "中高": 0.8, "中": 0.6, "低": 0.4, "无": 0.2}

        for tf_name, analysis in trend_analysis.items():
            if analysis["valid"]:
                trend = analysis["trend"]
                valid_trends[tf_name] = trend
                trend_counts[trend] += 1

                # 权重计算: 时间框架权重 * 趋势持续时间的平方根 * 置信度权重
                tf_weight = self.timeframes[tf_name]["weight"]
                duration_factor = np.sqrt(min(analysis["periods"], 10)) / 3  # 最多贡献权重的3倍
                conf_weight = confidence_weights.get(analysis["confidence"], 0.2)

                total_weight = tf_weight * duration_factor * conf_weight
                weighted_scores[trend] += total_weight

        # 计算趋势一致性
        total_valid = sum(trend_counts.values())
        if total_valid > 0:
            # 找出得分最高的趋势
            dominant_trend = max(weighted_scores, key=weighted_scores.get)
            highest_score = weighted_scores[dominant_trend]

            # 计算一致性得分 (0-100)
            total_score = sum(weighted_scores.values())
            if total_score > 0:
                coherence_score = (highest_score / total_score) * 100
            else:
                coherence_score = 0

            # 计算趋势一致比例
            trend_agreement = (trend_counts[dominant_trend] / total_valid) * 100

            # 确定主导时间框架
            dominant_tf = None
            highest_contribution = 0

            for tf_name, analysis in trend_analysis.items():
                if analysis["valid"] and analysis["trend"] == dominant_trend:
                    tf_weight = self.timeframes[tf_name]["weight"]
                    duration_factor = np.sqrt(min(analysis["periods"], 10)) / 3
                    conf_weight = confidence_weights.get(analysis["confidence"], 0.2)

                    contribution = tf_weight * duration_factor * conf_weight
                    if contribution > highest_contribution:
                        highest_contribution = contribution
                        dominant_tf = tf_name

            # 检测趋势冲突
            trend_conflicts = []
            if trend_counts["UP"] > 0 and trend_counts["DOWN"] > 0:
                # 收集具体冲突
                up_timeframes = [tf for tf, trend in valid_trends.items() if trend == "UP"]
                down_timeframes = [tf for tf, trend in valid_trends.items() if trend == "DOWN"]

                conflict_description = f"上升趋势({','.join(up_timeframes)}) vs 下降趋势({','.join(down_timeframes)})"
                trend_conflicts.append(conflict_description)

            # 确定一致性级别
            if coherence_score >= 80 and trend_agreement >= 80:
                agreement_level = "高度一致"
            elif coherence_score >= 70 and trend_agreement >= 60:
                agreement_level = "较强一致"
            elif coherence_score >= 60 and trend_agreement >= 50:
                agreement_level = "中等一致"
            elif coherence_score >= 50:
                agreement_level = "弱一致"
            else:
                agreement_level = "不一致"

            # 生成交易建议
            if dominant_trend == "UP" and agreement_level in ["高度一致", "较强一致"]:
                recommendation = "BUY"
            elif dominant_trend == "DOWN" and agreement_level in ["高度一致", "较强一致"]:
                recommendation = "SELL"
            elif dominant_trend != "NEUTRAL" and agreement_level == "中等一致":
                recommendation = f"LIGHT_{dominant_trend}"  # LIGHT_UP or LIGHT_DOWN
            else:
                recommendation = "NEUTRAL"

            # 更新结果
            result.update({
                "coherence_score": coherence_score,
                "trend_agreement": trend_agreement,
                "dominant_timeframe": dominant_tf,
                "dominant_trend": dominant_trend,
                "trend_conflicts": trend_conflicts,
                "agreement_level": agreement_level,
                "recommendation": recommendation,
                "weighted_scores": weighted_scores
            })

        # 打印结果
        agreement_color = (
            Colors.GREEN + Colors.BOLD if result["agreement_level"] == "高度一致" else
            Colors.GREEN if result["agreement_level"] == "较强一致" else
            Colors.YELLOW if result["agreement_level"] == "中等一致" else
            Colors.RED if result["agreement_level"] == "弱一致" else
            Colors.RED + Colors.BOLD
        )

        dominant_trend_color = (
            Colors.GREEN if result["dominant_trend"] == "UP" else
            Colors.RED if result["dominant_trend"] == "DOWN" else
            Colors.GRAY
        )

        print_colored("\n===== 时间框架一致性分析 =====", Colors.BLUE + Colors.BOLD)
        print_colored(
            f"一致性得分: {result['coherence_score']:.1f}/100, "
            f"趋势一致率: {result['trend_agreement']:.1f}%",
            Colors.INFO
        )
        print_colored(
            f"主导趋势: {dominant_trend_color}{result['dominant_trend']}{Colors.RESET}, "
            f"主导时间框架: {result['dominant_timeframe'] or '未知'}",
            Colors.INFO
        )
        print_colored(
            f"一致性级别: {agreement_color}{result['agreement_level']}{Colors.RESET}",
            Colors.INFO
        )

        if result["trend_conflicts"]:
            print_colored(f"趋势冲突: {', '.join(result['trend_conflicts'])}", Colors.WARNING)

        print_colored(
            f"交易建议: {result['recommendation']}",
            Colors.GREEN if "BUY" in result['recommendation'] else
            Colors.RED if "SELL" in result['recommendation'] else
            Colors.YELLOW
        )

        # 缓存结果
        self.coherence_cache[symbol] = {
            "result": result,
            "timestamp": time.time()
        }

        return result

    def get_timeframe_coherence(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """获取时间框架一致性分析，支持缓存

        参数:
            symbol: 交易对
            force_refresh: 是否强制刷新

        返回:
            一致性分析结果
        """
        cache_ttl = 300  # 缓存有效期5分钟
        current_time = time.time()

        # 检查缓存
        if not force_refresh and symbol in self.coherence_cache:
            cache_entry = self.coherence_cache[symbol]
            if (current_time - cache_entry["timestamp"]) < cache_ttl:
                print_colored(f"使用缓存的一致性分析结果 ({(current_time - cache_entry['timestamp']):.0f}秒前)",
                              Colors.CYAN)
                return cache_entry["result"]

        # 获取所有时间框架数据
        timeframe_data = self.fetch_all_timeframes(symbol, force_refresh)

        # 分析趋势
        trend_analysis = self.analyze_timeframe_trends(symbol, timeframe_data)

        # 计算一致性
        coherence_result = self.calculate_timeframe_coherence(symbol, trend_analysis)

        return coherence_result

    def detect_primary_timeframe(self, symbol: str, market_data: Dict[str, Any] = None) -> str:
        """检测当前市场的主导时间框架

        基于市场特征检测最适合当前交易的时间框架

        参数:
            symbol: 交易对
            market_data: 市场数据（可选）

        返回:
            主导时间框架
        """
        # 获取一致性分析
        coherence = self.get_timeframe_coherence(symbol)
        if coherence["dominant_timeframe"]:
            return coherence["dominant_timeframe"]

        # 如果一致性分析未能确定主导时间框架，使用波动性分析
        try:
            # 获取默认时间框架数据
            default_tf = "15m"
            if default_tf in self.timeframes and symbol in self.timeframes[default_tf]["data"]:
                df = self.timeframes[default_tf]["data"][symbol]

                if 'ATR' in df.columns:
                    # 计算ATR比率
                    atr = df['ATR'].iloc[-1]
                    atr_mean = df['ATR'].mean()
                    atr_ratio = atr / atr_mean if atr_mean > 0 else 1.0

                    # 根据波动性判断适合的时间框架
                    if atr_ratio > 2.0:  # 极端高波动
                        return "1h"  # 使用更高时间框架避免噪声
                    elif atr_ratio > 1.5:  # 高波动
                        return "15m"
                    elif atr_ratio < 0.5:  # 低波动
                        return "5m"  # 使用更低时间框架捕捉小波动
                    else:  # 中等波动
                        return "15m"

                # 检查ADX指标
                if 'ADX' in df.columns:
                    adx = df['ADX'].iloc[-1]
                    if adx > 30:  # 强趋势
                        return "1h"  # 高时间框架更适合强趋势
                    elif adx < 15:  # 弱趋势
                        return "5m"  # 低时间框架更适合弱趋势或震荡

            # 默认时间框架
            return "15m"
        except Exception as e:
            print_colored(f"❌ 检测主导时间框架失败: {e}", Colors.ERROR)
            return "15m"  # 默认时间框架

    def adjust_quality_score(self, symbol: str, original_score: float) -> Tuple[float, Dict[str, Any]]:
        """根据时间框架一致性调整质量评分

        参数:
            symbol: 交易对
            original_score: 原始质量评分

        返回:
            (调整后的质量评分, 调整明细)
        """
        # 获取一致性分析
        coherence = self.get_timeframe_coherence(symbol)

        # 初始化调整信息
        adjustment_info = {
            "original_score": original_score,
            "final_score": original_score,
            "adjustments": []
        }

        # 根据一致性进行调整
        if coherence["agreement_level"] == "高度一致":
            # 高度一致性加分
            adjustment = min(2.0, original_score * 0.2)  # 最多加2分或原分数的20%
            new_score = min(10.0, original_score + adjustment)
            adjustment_info["adjustments"].append({
                "reason": "高度时间框架一致性",
                "value": adjustment
            })
        elif coherence["agreement_level"] == "较强一致":
            # 较强一致性加分
            adjustment = min(1.0, original_score * 0.1)  # 最多加1分或原分数的10%
            new_score = min(10.0, original_score + adjustment)
            adjustment_info["adjustments"].append({
                "reason": "较强时间框架一致性",
                "value": adjustment
            })
        elif coherence["agreement_level"] == "不一致":
            # 不一致减分
            adjustment = min(2.0, original_score * 0.2)  # 最多减2分或原分数的20%
            new_score = max(0.0, original_score - adjustment)
            adjustment_info["adjustments"].append({
                "reason": "时间框架不一致",
                "value": -adjustment
            })
        else:
            # 中等或弱一致性不调整
            new_score = original_score
            adjustment_info["adjustments"].append({
                "reason": "中等或弱一致性，无调整",
                "value": 0
            })

        # 趋势冲突额外减分
        if coherence["trend_conflicts"]:
            conflict_penalty = min(1.0, original_score * 0.1)  # 最多减1分或原分数的10%
            new_score = max(0.0, new_score - conflict_penalty)
            adjustment_info["adjustments"].append({
                "reason": "时间框架趋势冲突",
                "value": -conflict_penalty
            })

        # 调整特定条件下的评分
        if coherence["dominant_trend"] == "UP" and original_score < 5.0:
            # 主导趋势是向上但原始评分较低，轻微加分使其接近中性
            adjustment = min(1.0, (5.0 - original_score) * 0.5)
            new_score = new_score + adjustment
            adjustment_info["adjustments"].append({
                "reason": "上升主导趋势但原始评分较低",
                "value": adjustment
            })
        elif coherence["dominant_trend"] == "DOWN" and original_score > 5.0:
            # 主导趋势是向下但原始评分较高，轻微减分使其接近中性
            adjustment = min(1.0, (original_score - 5.0) * 0.5)
            new_score = new_score - adjustment
            adjustment_info["adjustments"].append({
                "reason": "下降主导趋势但原始评分较高",
                "value": -adjustment
            })

        # 确保最终分数在0-10范围内
        new_score = max(0.0, min(10.0, new_score))
        adjustment_info["final_score"] = new_score

        # 打印调整结果
        print_colored("\n===== 质量评分调整 =====", Colors.BLUE + Colors.BOLD)
        print_colored(f"原始评分: {original_score:.2f}", Colors.INFO)

        for adj in adjustment_info["adjustments"]:
            if adj["value"] != 0:
                adj_color = Colors.GREEN if adj["value"] > 0 else Colors.RED
                print_colored(
                    f"{adj['reason']}: {adj_color}{adj['value']:+.2f}{Colors.RESET}",
                    Colors.INFO
                )

        print_colored(f"最终评分: {new_score:.2f}", Colors.INFO)

        return new_score, adjustment_info

    def generate_signal(self, symbol: str, quality_score: float) -> Tuple[str, float, Dict[str, Any]]:
        """基于多时间框架分析生成信号

        参数:
            symbol: 交易对
            quality_score: 质量评分

        返回:
            (信号, 调整后的质量评分, 详细信息)
        """
        # 获取一致性分析
        coherence = self.get_timeframe_coherence(symbol)

        # 调整质量评分
        adjusted_score, adjustment_info = self.adjust_quality_score(symbol, quality_score)

        # 确定信号
        if coherence["recommendation"] == "BUY" and adjusted_score >= 6.0:
            signal = "BUY"
        elif coherence["recommendation"] == "SELL" and adjusted_score <= 4.0:
            signal = "SELL"
        elif "LIGHT_UP" in coherence["recommendation"] and adjusted_score >= 5.5:
            signal = "LIGHT_BUY"  # 轻度买入信号
        elif "LIGHT_DOWN" in coherence["recommendation"] and adjusted_score <= 4.5:
            signal = "LIGHT_SELL"  # 轻度卖出信号
        else:
            signal = "NEUTRAL"

        # 详细信息
        details = {
            "coherence": coherence,
            "adjusted_score": adjusted_score,
            "adjustment_info": adjustment_info,
            "primary_timeframe": self.detect_primary_timeframe(symbol)
        }

        return signal, adjusted_score, details