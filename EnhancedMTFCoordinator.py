"""
增强版多时间框架分析模块 - 短期交易优化版
专注于1-5小时交易窗口，整合价格预测与市场结构分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from datetime import datetime, timedelta
from logger_utils import Colors, print_colored
from indicators_module import calculate_optimized_indicators, get_smc_trend_and_duration


class EnhancedMTFCoordinator:
    """增强版多时间框架协调类，专注于短期交易，确保信号与价格预测一致"""

    def __init__(self, client, logger=None):
        """初始化增强版多时间框架协调器

        参数:
            client: Binance客户端
            logger: 日志对象
        """
        self.client = client
        self.logger = logger

        # 定义交易相关时间框架 - 优先考虑短期时间框架
        self.timeframes = {
            "1m": {"interval": "1m", "weight": 0.5, "data": {}, "last_update": {}},
            "5m": {"interval": "5m", "weight": 0.8, "data": {}, "last_update": {}},
            "15m": {"interval": "15m", "weight": 1.0, "data": {}, "last_update": {}},
            "30m": {"interval": "30m", "weight": 1.2, "data": {}, "last_update": {}},  # 新增30分钟时间框架
            "1h": {"interval": "1h", "weight": 1.0, "data": {}, "last_update": {}},  # 降低权重
            "4h": {"interval": "4h", "weight": 0.7, "data": {}, "last_update": {}}  # 降低权重
        }

        # 更新间隔 - 更频繁地更新短期时间框架
        self.update_interval = {
            "1m": 30,  # 30秒
            "5m": 120,  # 2分钟
            "15m": 300,  # 5分钟
            "30m": 600,  # 10分钟
            "1h": 900,  # 15分钟
            "4h": 1800  # 30分钟
        }

        # 趋势一致性缓存
        self.coherence_cache = {}

        # 价格预测缓存
        self.price_prediction_cache = {}

        # 入场机会跟踪
        self.entry_opportunities = {}

        # 平仓建议跟踪
        self.exit_recommendations = {}

        print_colored("🔄 增强版多时间框架协调器初始化完成", Colors.GREEN)

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
                periods = self._minutes_to_periods(duration, tf_name)

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

    def _minutes_to_periods(self, minutes: int, timeframe: str) -> float:
        """将分钟转换为对应时间框架的周期数"""
        if timeframe == "1m":
            return minutes
        elif timeframe == "5m":
            return minutes / 5
        elif timeframe == "15m":
            return minutes / 15
        elif timeframe == "30m":
            return minutes / 30
        elif timeframe == "1h":
            return minutes / 60
        elif timeframe == "4h":
            return minutes / 240
        else:
            return minutes

    def predict_price_movement(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame],
                               horizon_minutes: int = 60) -> Dict[str, Any]:
        """基于多时间框架分析进行价格运动预测

        参数:
            symbol: 交易对
            timeframe_data: 各时间框架的DataFrame字典
            horizon_minutes: 预测时间范围（分钟）

        返回:
            价格预测结果
        """
        # 检查缓存
        cache_key = f"{symbol}_{horizon_minutes}"
        current_time = time.time()
        if cache_key in self.price_prediction_cache:
            cache_entry = self.price_prediction_cache[cache_key]
            # 缓存5分钟有效
            if current_time - cache_entry['timestamp'] < 300:
                print_colored(f"使用缓存的价格预测: {symbol}", Colors.CYAN)
                return cache_entry['prediction']

        print_colored(f"🔮 预测{symbol}在{horizon_minutes}分钟内的价格走势", Colors.BLUE)

        # 获取当前价格 - 使用最短时间框架数据
        short_tf = "1m" if "1m" in timeframe_data and not timeframe_data["1m"].empty else "5m"
        if short_tf not in timeframe_data or timeframe_data[short_tf].empty:
            print_colored(f"⚠️ 无法获取{symbol}当前价格", Colors.WARNING)
            return {"valid": False, "error": "无法获取当前价格"}

        current_price = timeframe_data[short_tf]['close'].iloc[-1]

        # 分别预测各时间框架
        predictions = {}
        total_weight = 0

        for tf_name, df in timeframe_data.items():
            if df.empty or len(df) < 10:
                continue

            # 计算线性回归斜率
            window_length = min(len(df), 20 if tf_name in ["1m", "5m"] else 15)
            window = df['close'].tail(window_length)
            x = np.arange(len(window))

            try:
                # 使用加权多项式拟合预测 - 对短期数据更敏感
                if len(window) >= 10:
                    # 为最近数据赋予更高权重
                    weights = np.linspace(0.5, 1.0, len(window))
                    # 对于短时间框架，使用更高阶多项式
                    poly_degree = 2 if tf_name in ["1m", "5m", "15m"] else 1
                    poly_fit = np.polyfit(x, window, poly_degree, w=weights)

                    # 预测
                    candles_needed = self._minutes_to_periods(horizon_minutes, tf_name)
                    if poly_degree == 1:
                        # 线性预测
                        slope, intercept = poly_fit
                        prediction = slope * (len(window) + candles_needed) + intercept
                    else:
                        # 多项式预测
                        prediction = np.polyval(poly_fit, len(window) + candles_needed)

                    # 计算置信度 - R²适应度
                    p = np.poly1d(poly_fit)
                    fitted = p(x)
                    mean = np.mean(window)
                    ss_tot = np.sum((window - mean) ** 2)
                    ss_res = np.sum((window - fitted) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    # 分配权重 - 短期时间框架更重要
                    tf_weight = self.timeframes[tf_name]["weight"] * (r2 + 0.2)  # 加上基础权重

                    predictions[tf_name] = {
                        "prediction": prediction,
                        "change_pct": (prediction - current_price) / current_price * 100,
                        "confidence": r2,
                        "weight": tf_weight
                    }

                    total_weight += tf_weight

                    # 显示预测
                    change_str = f"{predictions[tf_name]['change_pct']:+.2f}%"
                    change_color = Colors.GREEN if predictions[tf_name]['change_pct'] > 0 else Colors.RED
                    print_colored(
                        f"{tf_name}: 预测 {prediction:.6f} ({change_color}{change_str}{Colors.RESET}), "
                        f"R²: {r2:.3f}, 权重: {tf_weight:.2f}",
                        Colors.INFO
                    )
            except Exception as e:
                print_colored(f"❌ {tf_name}价格预测失败: {e}", Colors.ERROR)

        # 如果没有有效预测
        if not predictions or total_weight == 0:
            print_colored(f"⚠️ {symbol}没有有效的预测", Colors.WARNING)
            return {"valid": False, "error": "无有效预测"}

        # 计算加权平均预测
        weighted_prediction = sum(p["prediction"] * p["weight"] for p in predictions.values()) / total_weight
        weighted_change_pct = (weighted_prediction - current_price) / current_price * 100

        # 计算短期与中长期预测方向
        short_term_predictions = {k: v for k, v in predictions.items() if k in ["1m", "5m", "15m", "30m"]}
        long_term_predictions = {k: v for k, v in predictions.items() if k in ["1h", "4h"]}

        # 如果有足够的短期预测
        short_term_direction = None
        if short_term_predictions:
            short_term_weight = sum(p["weight"] for p in short_term_predictions.values())
            short_term_pred = sum(
                p["prediction"] * p["weight"] for p in short_term_predictions.values()) / short_term_weight
            short_term_change = (short_term_pred - current_price) / current_price * 100
            short_term_direction = "UP" if short_term_change > 0 else "DOWN"

        # 如果有足够的长期预测
        long_term_direction = None
        if long_term_predictions:
            long_term_weight = sum(p["weight"] for p in long_term_predictions.values())
            long_term_pred = sum(
                p["prediction"] * p["weight"] for p in long_term_predictions.values()) / long_term_weight
            long_term_change = (long_term_pred - current_price) / current_price * 100
            long_term_direction = "UP" if long_term_change > 0 else "DOWN"

        # 确定建议方向 - 对于短期交易，优先考虑短期方向
        suggested_direction = None
        direction_confidence = 0.0

        if short_term_direction and long_term_direction:
            if short_term_direction == long_term_direction:
                # 短期和长期方向一致
                suggested_direction = short_term_direction
                direction_confidence = 0.9  # 高置信度
            else:
                # 方向不一致，优先短期
                suggested_direction = short_term_direction
                direction_confidence = 0.6  # 中等置信度
        elif short_term_direction:
            suggested_direction = short_term_direction
            direction_confidence = 0.7  # 中高置信度
        elif long_term_direction:
            suggested_direction = long_term_direction
            direction_confidence = 0.5  # 中等置信度

        # 输出结果
        change_str = f"{weighted_change_pct:+.2f}%"
        change_color = Colors.GREEN if weighted_change_pct > 0 else Colors.RED

        print_colored(
            f"综合预测: {weighted_prediction:.6f} ({change_color}{change_str}{Colors.RESET}), "
            f"方向: {suggested_direction}, 置信度: {direction_confidence:.2f}",
            Colors.CYAN + Colors.BOLD
        )

        if short_term_direction and long_term_direction and short_term_direction != long_term_direction:
            print_colored(
                f"⚠️ 短期与长期预测方向不一致: 短期={short_term_direction}, 长期={long_term_direction}",
                Colors.YELLOW
            )

        # 创建结果
        result = {
            "valid": True,
            "current_price": current_price,
            "predicted_price": weighted_prediction,
            "change_pct": weighted_change_pct,
            "direction": suggested_direction,
            "confidence": direction_confidence,
            "short_term_direction": short_term_direction,
            "long_term_direction": long_term_direction,
            "timeframe_predictions": predictions,
            "conflict": short_term_direction != long_term_direction if short_term_direction and long_term_direction else False
        }

        # 缓存结果
        self.price_prediction_cache[cache_key] = {
            'prediction': result,
            'timestamp': current_time
        }

        return result

    def calculate_timeframe_coherence(self, symbol: str, trend_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算时间框架一致性 - 专注于短期交易目标

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
            "recommendation": "NEUTRAL",
            "short_term_bias": None,  # 短期偏向
            "short_term_confidence": 0.0  # 短期置信度
        }

        # 收集有效的趋势
        valid_trends = {}
        trend_counts = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        weighted_scores = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        confidence_weights = {"高": 1.0, "中高": 0.8, "中": 0.6, "低": 0.4, "无": 0.2}

        # 分离短期和长期时间框架
        short_term_tfs = ["1m", "5m", "15m", "30m"]
        long_term_tfs = ["1h", "4h"]

        # 短期和长期趋势权重
        short_term_weights = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        long_term_weights = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}

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

                # 添加到短期或长期权重
                if tf_name in short_term_tfs:
                    short_term_weights[trend] += total_weight
                else:
                    long_term_weights[trend] += total_weight

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

            # 确定短期偏向
            short_term_total = sum(short_term_weights.values())
            if short_term_total > 0:
                short_term_trend = max(short_term_weights, key=short_term_weights.get)
                short_term_confidence = short_term_weights[short_term_trend] / short_term_total

                # 如果短期趋势是NEUTRAL，选择次高的趋势
                if short_term_trend == "NEUTRAL" and (short_term_weights["UP"] > 0 or short_term_weights["DOWN"] > 0):
                    short_term_trend = "UP" if short_term_weights["UP"] > short_term_weights["DOWN"] else "DOWN"
                    short_term_confidence = short_term_weights[short_term_trend] / short_term_total
            else:
                short_term_trend = "NEUTRAL"
                short_term_confidence = 0.0

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

            # 专注于短期交易的建议
            # 检查短期与长期趋势是否一致
            long_term_total = sum(long_term_weights.values())
            long_term_trend = "NEUTRAL"
            if long_term_total > 0:
                long_term_trend = max(long_term_weights, key=long_term_weights.get)

                # 如果长期趋势是NEUTRAL，选择次高的趋势
                if long_term_trend == "NEUTRAL" and (long_term_weights["UP"] > 0 or long_term_weights["DOWN"] > 0):
                    long_term_trend = "UP" if long_term_weights["UP"] > long_term_weights["DOWN"] else "DOWN"

            # 生成交易建议 - 优先考虑短期趋势
            if short_term_trend == "UP" and short_term_confidence >= 0.7:
                recommendation = "BUY"
            elif short_term_trend == "DOWN" and short_term_confidence >= 0.7:
                recommendation = "SELL"
            elif short_term_trend != "NEUTRAL" and short_term_confidence >= 0.5:
                # 中等置信度的短期信号
                recommendation = f"LIGHT_{short_term_trend}"  # LIGHT_UP or LIGHT_DOWN
            else:
                recommendation = "NEUTRAL"

            # 当短期与长期趋势冲突时，降低建议强度
            if short_term_trend != "NEUTRAL" and long_term_trend != "NEUTRAL" and short_term_trend != long_term_trend:
                # 冲突情况下保持短期方向，但降级为轻仓位
                if recommendation == "BUY":
                    recommendation = "LIGHT_UP"
                elif recommendation == "SELL":
                    recommendation = "LIGHT_DOWN"

            # 更新结果
            result.update({
                "coherence_score": coherence_score,
                "trend_agreement": trend_agreement,
                "dominant_timeframe": dominant_tf,
                "dominant_trend": dominant_trend,
                "trend_conflicts": trend_conflicts,
                "agreement_level": agreement_level,
                "recommendation": recommendation,
                "weighted_scores": weighted_scores,
                "short_term_bias": short_term_trend,
                "short_term_confidence": short_term_confidence,
                "long_term_trend": long_term_trend
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

        # 打印短期偏向
        short_term_color = (
            Colors.GREEN if result["short_term_bias"] == "UP" else
            Colors.RED if result["short_term_bias"] == "DOWN" else
            Colors.GRAY
        )

        print_colored(
            f"短期偏向: {short_term_color}{result['short_term_bias']}{Colors.RESET}, "
            f"置信度: {result['short_term_confidence']:.2f}",
            Colors.INFO
        )

        rec_color = (
            Colors.GREEN if "BUY" in result['recommendation'] else
            Colors.RED if "SELL" in result['recommendation'] else
            Colors.YELLOW
        )

        print_colored(
            f"交易建议: {rec_color}{result['recommendation']}{Colors.RESET}",
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

    def generate_signal(self, symbol: str, quality_score: float) -> Tuple[str, float, Dict[str, Any]]:
        """基于多时间框架分析和价格预测生成一致性交易信号

        参数:
            symbol: 交易对
            quality_score: 初步质量评分

        返回:
            (信号, 调整后的质量评分, 详细信息)
        """
        # 获取时间框架数据
        timeframe_data = self.fetch_all_timeframes(symbol, force_refresh=True)

        # 分析趋势
        trend_analysis = self.analyze_timeframe_trends(symbol, timeframe_data)

        # 计算一致性
        coherence = self.calculate_timeframe_coherence(symbol, trend_analysis)

        # 短期价格预测 - 60分钟内
        price_pred = self.predict_price_movement(symbol, timeframe_data, 60)

        # 计算支撑阻力位
        support_resistance = self._calculate_support_resistance(symbol, timeframe_data)

        # 调整质量评分
        adjusted_score, adjustment_info = self.adjust_quality_score(symbol, quality_score, coherence, price_pred)

        # 生成交易信号 - 确保预测与信号方向一致
        signal = "NEUTRAL"
        final_info = {}

        if price_pred.get("valid", False):
            # 获取价格预测方向
            pred_direction = price_pred.get("direction")
            pred_confidence = price_pred.get("confidence", 0)

            # 获取建议信号
            suggested_signal = coherence.get("recommendation", "NEUTRAL")

            # 检查预测方向与信号方向是否一致
            if pred_direction == "UP" and "BUY" in suggested_signal:
                # 上涨预测，买入信号
                signal = suggested_signal

                # 如果预测置信度高，可以提升信号强度
                if pred_confidence > 0.8 and signal.startswith("LIGHT_"):
                    signal = "BUY"  # 提升为全仓位信号

            elif pred_direction == "DOWN" and "SELL" in suggested_signal:
                # 下跌预测，卖出信号
                signal = suggested_signal

                # 如果预测置信度高，可以提升信号强度
                if pred_confidence > 0.8 and signal.startswith("LIGHT_"):
                    signal = "SELL"  # 提升为全仓位信号

            elif pred_direction and suggested_signal != "NEUTRAL":
                # 方向不一致，但有明确信号和预测
                if pred_confidence > 0.8:
                    # 如果预测置信度高，优先考虑预测方向
                    signal = "BUY" if pred_direction == "UP" else "SELL"
                    signal = "LIGHT_" + signal.replace("LIGHT_", "")  # 降级为轻仓位

                    print_colored(
                        f"⚠️ 预测方向({pred_direction})与建议信号({suggested_signal})不一致，"
                        f"但预测置信度高({pred_confidence:.2f})，采用预测方向",
                        Colors.WARNING
                    )
                else:
                    # 置信度不高，等待入场
                    signal = "NEUTRAL"
                    print_colored(
                        f"⚠️ 预测方向({pred_direction})与建议信号({suggested_signal})不一致，"
                        f"且预测置信度不高({pred_confidence:.2f})，保持观望",
                        Colors.WARNING
                    )
            else:
                # 没有明确方向，保持中性
                signal = "NEUTRAL"
                print_colored(f"没有明确方向，保持观望", Colors.YELLOW)
        else:
            # 无效预测，使用趋势一致性建议
            signal = coherence.get("recommendation", "NEUTRAL")

        # 当信号与预测方向不一致时，并且有明确方向时，转为LIGHT或NEUTRAL
        if price_pred.get("valid", False) and coherence.get("recommendation", "NEUTRAL") != "NEUTRAL":
            rec_direction = "UP" if "BUY" in coherence["recommendation"] else "DOWN" if "SELL" in coherence[
                "recommendation"] else None
            pred_direction = price_pred.get("direction")

            if rec_direction and pred_direction and rec_direction != pred_direction:
                # 信号与预测方向不一致
                if not signal.startswith("LIGHT_") and signal != "NEUTRAL":
                    print_colored(
                        f"⚠️ 降级信号强度: 趋势方向({rec_direction})与预测方向({pred_direction})不一致",
                        Colors.WARNING
                    )
                    if signal == "BUY":
                        signal = "LIGHT_UP"
                    elif signal == "SELL":
                        signal = "LIGHT_DOWN"

        # 探测入场机会 - 如果有更好的入场时机，建议等待
        entry_opportunity = self._detect_entry_opportunity(symbol, signal, timeframe_data, support_resistance,
                                                           price_pred)
        if entry_opportunity["recommendation"] == "WAIT" and signal != "NEUTRAL":
            # 有更好的入场时机
            prev_signal = signal
            signal = "NEUTRAL"  # 暂时不交易

            print_colored(
                f"⏳ 建议等待更好的入场点: {entry_opportunity['reason']}",
                Colors.YELLOW
            )

            # 记录入场机会
            self.entry_opportunities[symbol] = {
                "original_signal": prev_signal,
                "target_price": entry_opportunity.get("target_price"),
                "expiry_time": time.time() + 3600,  # 1小时过期
                "quality_score": adjusted_score,
                "reason": entry_opportunity['reason']
            }

        # 构建详细信息
        final_info = {
            "coherence": coherence,
            "price_prediction": price_pred,
            "support_resistance": support_resistance,
            "adjustment_info": adjustment_info,
            "primary_timeframe": coherence.get("dominant_timeframe", "15m"),
            "entry_opportunity": entry_opportunity
        }

        # 打印结果
        signal_color = (
            Colors.GREEN if signal == "BUY" else
            Colors.GREEN + Colors.BOLD if signal == "LIGHT_UP" else
            Colors.RED if signal == "SELL" else
            Colors.RED + Colors.BOLD if signal == "LIGHT_DOWN" else
            Colors.GRAY
        )

        print_colored(
            f"\n===== 最终交易信号 =====",
            Colors.BLUE + Colors.BOLD
        )
        print_colored(
            f"信号: {signal_color}{signal}{Colors.RESET}",
            Colors.INFO
        )
        print_colored(
            f"质量评分: {quality_score:.2f} -> {adjusted_score:.2f}",
            Colors.INFO
        )

        if price_pred.get("valid", False):
            change_str = f"{price_pred['change_pct']:+.2f}%"
            change_color = Colors.GREEN if price_pred['change_pct'] > 0 else Colors.RED
            print_colored(
                f"价格预测: {price_pred['predicted_price']:.6f} ({change_color}{change_str}{Colors.RESET})",
                Colors.INFO
            )

        return signal, adjusted_score, final_info

    def adjust_quality_score(self, symbol: str, original_score: float,
                             coherence: Dict[str, Any], price_pred: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """根据时间框架一致性和价格预测调整质量评分

        参数:
            symbol: 交易对
            original_score: 原始质量评分
            coherence: 一致性分析结果
            price_pred: 价格预测结果

        返回:
            (调整后的质量评分, 调整明细)
        """
        # 初始化调整信息
        adjustment_info = {
            "original_score": original_score,
            "final_score": original_score,
            "adjustments": []
        }

        # 1. 根据一致性进行调整
        if coherence["agreement_level"] == "高度一致":
            # 高度一致性加分
            adjustment = min(1.0, original_score * 0.1)  # 最多加1分或原分数的10%
            new_score = min(10.0, original_score + adjustment)
            adjustment_info["adjustments"].append({
                "reason": "高度时间框架一致性",
                "value": adjustment
            })
        elif coherence["agreement_level"] == "较强一致":
            # 较强一致性加分
            adjustment = min(0.5, original_score * 0.05)  # 最多加0.5分或原分数的5%
            new_score = min(10.0, original_score + adjustment)
            adjustment_info["adjustments"].append({
                "reason": "较强时间框架一致性",
                "value": adjustment
            })
        elif coherence["agreement_level"] == "不一致":
            # 不一致减分
            adjustment = min(1.0, original_score * 0.1)  # 最多减1分或原分数的10%
            new_score = max(0.0, original_score - adjustment)
            adjustment_info["adjustments"].append({
                "reason": "时间框架不一致",
                "value": -adjustment
            })
        else:
            # 中等或弱一致性小幅调整
            new_score = original_score
            adjustment_info["adjustments"].append({
                "reason": "中等或弱一致性，小幅调整",
                "value": 0
            })

        # 2. 根据价格预测调整
        if price_pred.get("valid", False):
            pred_direction = price_pred.get("direction")
            pred_confidence = price_pred.get("confidence", 0)
            pred_change_pct = price_pred.get("change_pct", 0)

            # 如果预测看涨但分数低，或预测看跌但分数高，进行调整
            if pred_direction == "UP" and original_score < 6.0:
                # 上涨预测但质量评分较低，适当提升
                adjustment = min(0.8, pred_confidence * 0.8)
                new_score = new_score + adjustment
                adjustment_info["adjustments"].append({
                    "reason": f"预测上涨({pred_change_pct:+.2f}%)但原始评分较低",
                    "value": adjustment
                })
            elif pred_direction == "DOWN" and original_score > 4.0:
                # 下跌预测但质量评分较高，适当降低
                adjustment = min(0.8, pred_confidence * 0.8)
                new_score = new_score - adjustment
                adjustment_info["adjustments"].append({
                    "reason": f"预测下跌({pred_change_pct:+.2f}%)但原始评分较高",
                    "value": -adjustment
                })

            # 预测幅度较大时的额外调整
            if abs(pred_change_pct) > 2.0:
                magnitude_factor = min(0.5, abs(pred_change_pct) * 0.1)

                if pred_direction == "UP":
                    # 大幅上涨预测，提升评分
                    new_score = new_score + magnitude_factor
                    adjustment_info["adjustments"].append({
                        "reason": f"预测大幅上涨({pred_change_pct:+.2f}%)",
                        "value": magnitude_factor
                    })
                else:
                    # 大幅下跌预测，降低评分
                    new_score = new_score - magnitude_factor
                    adjustment_info["adjustments"].append({
                        "reason": f"预测大幅下跌({pred_change_pct:+.2f}%)",
                        "value": -magnitude_factor
                    })

        # 3. 调整趋势与预测方向不一致的情况
        if price_pred.get("valid", False) and coherence.get("dominant_trend") != "NEUTRAL":
            dom_trend = coherence.get("dominant_trend")
            pred_direction = price_pred.get("direction")

            if dom_trend and pred_direction and dom_trend != pred_direction:
                # 趋势方向与预测方向不一致，降低评分置信度
                adjustment = min(1.5, original_score * 0.15)
                new_score = new_score - adjustment
                adjustment_info["adjustments"].append({
                    "reason": f"趋势方向({dom_trend})与预测方向({pred_direction})不一致",
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

    def _calculate_support_resistance(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """计算支撑位和阻力位

        参数:
            symbol: 交易对
            timeframe_data: 时间框架数据

        返回:
            支撑阻力位信息
        """
        # 初始化结果
        result = {
            "supports": [],
            "resistances": [],
            "nearest_support": None,
            "nearest_resistance": None
        }

        try:
            # 获取当前价格 - 使用最短时间框架
            short_tf = "1m" if "1m" in timeframe_data and not timeframe_data["1m"].empty else "5m"
            if short_tf not in timeframe_data or timeframe_data[short_tf].empty:
                return result

            current_price = timeframe_data[short_tf]['close'].iloc[-1]

            # 收集各时间框架的支撑阻力位
            all_supports = []
            all_resistances = []

            # 优先考虑较短时间框架的支撑阻力位
            priority_tfs = ["15m", "30m", "1h", "5m", "1m", "4h"]

            for tf_name in priority_tfs:
                if tf_name not in timeframe_data or timeframe_data[tf_name].empty:
                    continue

                df = timeframe_data[tf_name]

                # 1. 查找摆动高点和低点
                try:
                    from indicators_module import find_swing_points
                    swing_highs, swing_lows = find_swing_points(df)

                    # 区分支撑位和阻力位
                    current_supports = [low for low in swing_lows if low < current_price]
                    current_resistances = [high for high in swing_highs if high > current_price]

                    # 添加权重 - 短时间框架权重较低
                    weight = 0.6 if tf_name in ["1m", "5m"] else 1.0

                    for support in current_supports:
                        all_supports.append({
                            "price": support,
                            "type": "swing_low",
                            "timeframe": tf_name,
                            "weight": weight,
                            "distance": (current_price - support) / current_price
                        })

                    for resistance in current_resistances:
                        all_resistances.append({
                            "price": resistance,
                            "type": "swing_high",
                            "timeframe": tf_name,
                            "weight": weight,
                            "distance": (resistance - current_price) / current_price
                        })
                except Exception as e:
                    print_colored(f"❌ 计算{tf_name}摆动点失败: {e}", Colors.ERROR)

                # 2. 使用布林带作为支撑阻力
                if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
                    bb_lower = df['BB_Lower'].iloc[-1]
                    bb_upper = df['BB_Upper'].iloc[-1]

                    if bb_lower < current_price:
                        all_supports.append({
                            "price": bb_lower,
                            "type": "bollinger_lower",
                            "timeframe": tf_name,
                            "weight": 0.8,
                            "distance": (current_price - bb_lower) / current_price
                        })

                    if bb_upper > current_price:
                        all_resistances.append({
                            "price": bb_upper,
                            "type": "bollinger_upper",
                            "timeframe": tf_name,
                            "weight": 0.8,
                            "distance": (bb_upper - current_price) / current_price
                        })

                # 3. 使用EMA作为支撑阻力
                if 'EMA20' in df.columns:
                    ema20 = df['EMA20'].iloc[-1]

                    if ema20 < current_price:
                        all_supports.append({
                            "price": ema20,
                            "type": "ema20",
                            "timeframe": tf_name,
                            "weight": 0.7,
                            "distance": (current_price - ema20) / current_price
                        })
                    elif ema20 > current_price:
                        all_resistances.append({
                            "price": ema20,
                            "type": "ema20",
                            "timeframe": tf_name,
                            "weight": 0.7,
                            "distance": (ema20 - current_price) / current_price
                        })

            # 合并和过滤支撑阻力位
            # 1. 按价格排序
            all_supports.sort(key=lambda x: x["price"], reverse=True)  # 从高到低
            all_resistances.sort(key=lambda x: x["price"])  # 从低到高

            # 2. 合并接近的支撑阻力位
            merged_supports = []
            merged_resistances = []

            # 价格接近阈值 - 0.5%
            threshold = 0.005

            # 合并支撑位
            for support in all_supports:
                # 检查是否接近现有支撑位
                found_close = False
                for i, merged in enumerate(merged_supports):
                    if abs(support["price"] - merged["price"]) / merged["price"] < threshold:
                        # 合并 - 取加权平均
                        total_weight = merged["weight"] + support["weight"]
                        new_price = (merged["price"] * merged["weight"] + support["price"] * support[
                            "weight"]) / total_weight

                        # 更新合并后的支撑位
                        merged_supports[i] = {
                            "price": new_price,
                            "type": f"{merged['type']},{support['type']}",
                            "timeframe": f"{merged['timeframe']},{support['timeframe']}",
                            "weight": total_weight,
                            "distance": (current_price - new_price) / current_price
                        }
                        found_close = True
                        break

                if not found_close:
                    merged_supports.append(support)

            # 合并阻力位
            for resistance in all_resistances:
                # 检查是否接近现有阻力位
                found_close = False
                for i, merged in enumerate(merged_resistances):
                    if abs(resistance["price"] - merged["price"]) / merged["price"] < threshold:
                        # 合并 - 取加权平均
                        total_weight = merged["weight"] + resistance["weight"]
                        new_price = (merged["price"] * merged["weight"] + resistance["price"] * resistance[
                            "weight"]) / total_weight

                        # 更新合并后的阻力位
                        merged_resistances[i] = {
                            "price": new_price,
                            "type": f"{merged['type']},{resistance['type']}",
                            "timeframe": f"{merged['timeframe']},{resistance['timeframe']}",
                            "weight": total_weight,
                            "distance": (new_price - current_price) / current_price
                        }
                        found_close = True
                        break

                if not found_close:
                    merged_resistances.append(resistance)

            # 3. 按距离排序
            merged_supports.sort(key=lambda x: x["distance"])
            merged_resistances.sort(key=lambda x: x["distance"])

            # 获取最近的支撑阻力位
            nearest_support = merged_supports[0]["price"] if merged_supports else None
            nearest_resistance = merged_resistances[0]["price"] if merged_resistances else None

            # 创建结果
            result = {
                "supports": [s["price"] for s in merged_supports],
                "resistances": [r["price"] for r in merged_resistances],
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "detailed_supports": merged_supports,
                "detailed_resistances": merged_resistances,
                "current_price": current_price
            }

            # 输出结果
            print_colored("\n===== 支撑阻力位分析 =====", Colors.BLUE + Colors.BOLD)
            print_colored(f"当前价格: {current_price:.6f}", Colors.INFO)

            if nearest_support:
                support_distance = (current_price - nearest_support) / current_price * 100
                print_colored(f"最近支撑位: {nearest_support:.6f} (距离: {support_distance:.2f}%)", Colors.INFO)

            if nearest_resistance:
                resistance_distance = (nearest_resistance - current_price) / current_price * 100
                print_colored(f"最近阻力位: {nearest_resistance:.6f} (距离: {resistance_distance:.2f}%)", Colors.INFO)

            return result

        except Exception as e:
            print_colored(f"❌ 计算支撑阻力位失败: {e}", Colors.ERROR)
            return result

    def _detect_entry_opportunity(self, symbol: str, signal: str, timeframe_data: Dict[str, pd.DataFrame],
                                  support_resistance: Dict[str, Any], price_pred: Dict[str, Any]) -> Dict[str, Any]:
        """检测更好的入场机会

        参数:
            symbol: 交易对
            signal: 交易信号
            timeframe_data: 时间框架数据
            support_resistance: 支撑阻力位信息
            price_pred: 价格预测信息

        返回:
            入场机会分析结果
        """
        result = {
            "recommendation": "PROCEED",  # PROCEED, WAIT
            "reason": "可以立即入场",
            "target_price": None,
            "expected_minutes": 0
        }

        # 如果信号是NEUTRAL，无需检查入场机会
        if signal == "NEUTRAL":
            return result

        try:
            # 获取当前价格
            short_tf = "1m" if "1m" in timeframe_data and not timeframe_data["1m"].empty else "5m"
            if short_tf not in timeframe_data or timeframe_data[short_tf].empty:
                return result

            current_price = timeframe_data[short_tf]['close'].iloc[-1]

            # 获取支撑阻力位
            nearest_support = support_resistance.get("nearest_support")
            nearest_resistance = support_resistance.get("nearest_resistance")

            # 获取价格预测
            pred_direction = price_pred.get("direction") if price_pred.get("valid", False) else None
            predicted_price = price_pred.get("predicted_price") if price_pred.get("valid", False) else None

            # 1. 检查价格是否接近支撑位或阻力位
            if signal in ["BUY", "LIGHT_UP"]:
                # 买入信号 - 检查是否应该等待回调到支撑位
                if nearest_support:
                    support_distance = (current_price - nearest_support) / current_price * 100

                    # 如果预测看涨但非常接近支撑位，可以直接入场
                    if pred_direction == "UP" and support_distance < 0.5:
                        result["recommendation"] = "PROCEED"
                        result["reason"] = f"价格接近支撑位({nearest_support:.6f})且预测上涨，适合立即入场"
                    # 如果预测看跌且支撑位不远，可以等待回调
                    elif pred_direction == "DOWN" and support_distance < 3.0:
                        result["recommendation"] = "WAIT"
                        result["reason"] = f"预测价格可能回调至支撑位附近，建议等待"
                        result["target_price"] = nearest_support * 1.005  # 略高于支撑位

                        # 估计等待时间
                        if predicted_price:
                            # 根据预测下跌速度估计时间
                            change_pct = (predicted_price - current_price) / current_price * 100
                            time_horizon = price_pred.get("time_horizon", 60)  # 默认60分钟

                            if change_pct < 0:  # 预测下跌
                                target_change = (result["target_price"] - current_price) / current_price * 100
                                estimated_minutes = int(
                                    min(120, max(10, target_change / change_pct * time_horizon * -1)))
                                result["expected_minutes"] = estimated_minutes
                    # 如果已经距离支撑位较远，考虑布林带位置
                    elif "BB_Middle" in timeframe_data["15m"].columns:
                        bb_middle = timeframe_data["15m"]['BB_Middle'].iloc[-1]
                        if current_price > bb_middle * 1.01:  # 价格显著高于中轨
                            result["recommendation"] = "WAIT"
                            result["reason"] = f"价格高于布林带中轨，可能有回调，建议等待"
                            result["target_price"] = bb_middle
                            result["expected_minutes"] = 30  # 预计30分钟

            elif signal in ["SELL", "LIGHT_DOWN"]:
                # 卖出信号 - 检查是否应该等待反弹到阻力位
                if nearest_resistance:
                    resistance_distance = (nearest_resistance - current_price) / current_price * 100

                    # 如果预测看跌但非常接近阻力位，可以直接入场
                    if pred_direction == "DOWN" and resistance_distance < 0.5:
                        result["recommendation"] = "PROCEED"
                        result["reason"] = f"价格接近阻力位({nearest_resistance:.6f})且预测下跌，适合立即入场"
                    # 如果预测看涨且阻力位不远，可以等待反弹
                    elif pred_direction == "UP" and resistance_distance < 3.0:
                        result["recommendation"] = "WAIT"
                        result["reason"] = f"预测价格可能反弹至阻力位附近，建议等待"
                        result["target_price"] = nearest_resistance * 0.995  # 略低于阻力位

                        # 估计等待时间
                        if predicted_price:
                            # 根据预测上涨速度估计时间
                            change_pct = (predicted_price - current_price) / current_price * 100
                            time_horizon = price_pred.get("time_horizon", 60)  # 默认60分钟

                            if change_pct > 0:  # 预测上涨
                                target_change = (result["target_price"] - current_price) / current_price * 100
                                estimated_minutes = int(min(120, max(10, target_change / change_pct * time_horizon)))
                                result["expected_minutes"] = estimated_minutes
                    # 如果已经距离阻力位较远，考虑布林带位置
                    elif "BB_Middle" in timeframe_data["15m"].columns:
                        bb_middle = timeframe_data["15m"]['BB_Middle'].iloc[-1]
                        if current_price < bb_middle * 0.99:  # 价格显著低于中轨
                            result["recommendation"] = "WAIT"
                            result["reason"] = f"价格低于布林带中轨，可能有反弹，建议等待"
                            result["target_price"] = bb_middle
                            result["expected_minutes"] = 30  # 预计30分钟

            # 2. 检查短期价格震荡情况
            if "1m" in timeframe_data and not timeframe_data["1m"].empty:
                df_1m = timeframe_data["1m"]
                if len(df_1m) >= 10:
                    # 计算短期震荡程度
                    recent_highs = df_1m['high'].tail(10).max()
                    recent_lows = df_1m['low'].tail(10).min()
                    volatility = (recent_highs - recent_lows) / current_price * 100

                    if volatility > 1.0:  # 超过1%的短期波动
                        # 对于买入信号，建议在低点入场
                        if signal in ["BUY", "LIGHT_UP"]:
                            avg_price = df_1m['close'].tail(5).mean()
                            if current_price > avg_price * 1.005:  # 当前价格高于5分钟均价0.5%
                                result["recommendation"] = "WAIT"
                                result["reason"] = f"短期波动较大({volatility:.2f}%)且价格高于短期均价，建议等待回调"
                                result["target_price"] = avg_price
                                result["expected_minutes"] = 10  # 预计10分钟内回调

                        # 对于卖出信号，建议在高点入场
                        elif signal in ["SELL", "LIGHT_DOWN"]:
                            avg_price = df_1m['close'].tail(5).mean()
                            if current_price < avg_price * 0.995:  # 当前价格低于5分钟均价0.5%
                                result["recommendation"] = "WAIT"
                                result["reason"] = f"短期波动较大({volatility:.2f}%)且价格低于短期均价，建议等待反弹"
                                result["target_price"] = avg_price
                                result["expected_minutes"] = 10  # 预计10分钟内反弹

            # 3. 价格预测与入场时机协调
            if price_pred.get("valid", False) and predicted_price:
                # 如果预测方向与信号方向一致，且变化幅度超过2%
                change_pct = (predicted_price - current_price) / current_price * 100

                if signal in ["BUY", "LIGHT_UP"] and pred_direction == "UP" and change_pct > 2.0:
                    # 大幅上涨预期，建议立即入场
                    result["recommendation"] = "PROCEED"
                    result["reason"] = f"预测价格将大幅上涨({change_pct:.2f}%)，建议立即入场"

                elif signal in ["SELL", "LIGHT_DOWN"] and pred_direction == "DOWN" and change_pct < -2.0:
                    # 大幅下跌预期，建议立即入场
                    result["recommendation"] = "PROCEED"
                    result["reason"] = f"预测价格将大幅下跌({change_pct:.2f}%)，建议立即入场"

            # 打印结果
            rec_color = Colors.GREEN if result["recommendation"] == "PROCEED" else Colors.YELLOW
            print_colored(
                f"入场分析: {rec_color}{result['recommendation']}{Colors.RESET} - {result['reason']}",
                Colors.INFO
            )

            if result["target_price"]:
                print_colored(
                    f"目标价格: {result['target_price']:.6f}, 预计等待时间: {result['expected_minutes']}分钟",
                    Colors.INFO
                )

            return result

        except Exception as e:
            print_colored(f"❌ 检测入场机会失败: {e}", Colors.ERROR)
            return result

    def calculate_exit_points(self, symbol: str, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """计算最佳出场点，专注于短期获利

        参数:
            symbol: 交易对
            position_info: 持仓信息

        返回:
            出场点分析结果
        """
        # 获取所有时间框架数据
        timeframe_data = self.fetch_all_timeframes(symbol)

        # 分析趋势
        trend_analysis = self.analyze_timeframe_trends(symbol, timeframe_data)

        # 价格预测
        price_pred = self.predict_price_movement(symbol, timeframe_data, 60)

        # 支撑阻力位分析
        support_resistance = self._calculate_support_resistance(symbol, timeframe_data)

        # 获取当前价格
        current_price = support_resistance.get("current_price", 0)
        if current_price == 0 and "1m" in timeframe_data and not timeframe_data["1m"].empty:
            current_price = timeframe_data["1m"]['close'].iloc[-1]

        # 获取持仓方向
        position_side = position_info.get("position_side", "LONG")
        is_long = position_side in ["LONG", "BOTH"]

        # 获取入场价格
        entry_price = position_info.get("entry_price", current_price)

        # 计算当前利润
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100

        # 计算出场机会
        opportunities = self._calculate_exit_points(current_price, price_pred, support_resistance, trend_analysis)

        # 过滤方向匹配的机会
        matching_opportunities = []
        for opp in opportunities.get("opportunities", []):
            if (is_long and opp["side"] == "SELL") or (not is_long and opp["side"] == "BUY"):
                matching_opportunities.append(opp)

        # 按优先级排序
        if matching_opportunities:
            matching_opportunities.sort(key=lambda x: x.get("priority", 999))

        # 构建结果
        result = {
            "should_exit": len(matching_opportunities) > 0,
            "opportunities": matching_opportunities,
            "best_opportunity": matching_opportunities[0] if matching_opportunities else None,
            "current_price": current_price,
            "entry_price": entry_price,
            "profit_pct": profit_pct,
            "position_side": position_side
        }

        # 打印结果
        print_colored("\n===== 出场分析 =====", Colors.BLUE + Colors.BOLD)
        print_colored(
            f"持仓方向: {position_side}, 入场价: {entry_price:.6f}, 当前价: {current_price:.6f}",
            Colors.INFO
        )
        print_colored(
            f"当前利润: {Colors.GREEN if profit_pct > 0 else Colors.RED}{profit_pct:.2f}%{Colors.RESET}",
            Colors.INFO
        )

        if result["should_exit"]:
            best = result["best_opportunity"]
            print_colored(f"建议出场: {best['reason']}", Colors.GREEN + Colors.BOLD)
            print_colored(f"出场方式: {best['type']}单, 价格: {best['price']:.6f}", Colors.INFO)
        else:
            print_colored("暂无明确出场信号，建议持仓", Colors.YELLOW)

        return result

    def _calculate_exit_points(self, current_price: float, price_pred: Dict[str, Any],
                               support_resistance: Dict[str, Any], trend_analysis: Dict[str, Dict[str, Any]]) -> Dict[
        str, Any]:
        """计算最佳出场点，专注于短期获利

        参数:
            current_price: 当前价格
            price_pred: 价格预测结果
            support_resistance: 支撑阻力位分析结果
            trend_analysis: 趋势分析结果

        返回:
            出场机会分析结果
        """
        result = {
            "opportunities": [],
            "best_opportunity": None
        }

        try:
            # 提取预测方向
            direction = price_pred.get("direction") if price_pred.get("valid", False) else None
            predicted_price = price_pred.get("predicted_price") if price_pred.get("valid", False) else None
            prediction_confidence = price_pred.get("confidence", 0) if price_pred.get("valid", False) else 0

            # 提取短期趋势
            short_term_trends = {k: v for k, v in trend_analysis.items() if k in ["1m", "5m", "15m", "30m"]}
            short_term_directions = {}

            for tf, analysis in short_term_trends.items():
                if analysis["valid"]:
                    short_term_directions[tf] = analysis["trend"]

            # 计算短期趋势变化
            up_count = sum(1 for d in short_term_directions.values() if d == "UP")
            down_count = sum(1 for d in short_term_directions.values() if d == "DOWN")

            trend_shifting_down = False
            trend_shifting_up = False

            if "1m" in short_term_directions and "5m" in short_term_directions and "15m" in short_term_directions:
                # 趋势正在转向下降
                if (short_term_directions["1m"] == "DOWN" and
                        short_term_directions["5m"] in ["UP", "NEUTRAL"] and
                        short_term_directions["15m"] == "UP"):
                    trend_shifting_down = True

                # 趋势正在转向上升
                if (short_term_directions["1m"] == "UP" and
                        short_term_directions["5m"] in ["DOWN", "NEUTRAL"] and
                        short_term_directions["15m"] == "DOWN"):
                    trend_shifting_up = True

            # 基于支撑位阻力位和价格预测的出场机会
            opportunities = []

            # === 针对多头（BUY）持仓的出场策略 ===

            # 1. 价格接近阻力位 - 多头平仓机会
            if support_resistance["nearest_resistance"] is not None:
                resistance_price = support_resistance["nearest_resistance"]
                distance_pct = (resistance_price - current_price) / current_price * 100

                if distance_pct < 1.5:  # 非常接近阻力位
                    opportunity = {
                        "side": "SELL",  # 平多
                        "price": resistance_price * 0.995,  # 略低于阻力位
                        "type": "LIMIT",
                        "priority": 2,
                        "reason": f"价格接近阻力位 {resistance_price:.6f} (距离: {distance_pct:.2f}%)"
                    }
                    opportunities.append(opportunity)

            # 2. 趋势转向或价格预测看跌 - 多头平仓机会
            if trend_shifting_down or direction == "DOWN":
                reason = "趋势转向下降" if trend_shifting_down else "价格预测看跌"
                if predicted_price and predicted_price < current_price:
                    reason += f", 预测价格下跌至 {predicted_price:.6f}"

                # 如果预测置信度高或趋势转向明显，使用市价单
                if prediction_confidence > 0.7 or trend_shifting_down:
                    opportunity = {
                        "side": "SELL",  # 平多
                        "price": current_price,
                        "type": "MARKET",
                        "priority": 1,
                        "reason": reason
                    }
                else:
                    # 否则使用限价单
                    target_price = current_price * 1.005  # 略高于当前价格
                    opportunity = {
                        "side": "SELL",  # 平多
                        "price": target_price,
                        "type": "LIMIT",
                        "priority": 3,
                        "reason": f"{reason}, 等待轻微上涨至 {target_price:.6f}"
                    }
                opportunities.append(opportunity)

            # 3. 设置获利目标 - 多头平仓机会
            if predicted_price and predicted_price > current_price:
                gain_pct = (predicted_price - current_price) / current_price * 100

                # 针对1-5小时交易，设置适当的获利目标
                if gain_pct >= 3.0:  # 超过3%的获利机会
                    # 在预测价格和当前价格之间设置目标
                    target_price = current_price * (1 + min(gain_pct, 5.0) / 100 * 0.7)  # 取预测收益的70%
                    opportunity = {
                        "side": "SELL",  # 平多
                        "price": target_price,
                        "type": "LIMIT",
                        "priority": 2,
                        "reason": f"设置获利目标，预计上涨 {gain_pct:.2f}%，取 70% 收益"
                    }
                    opportunities.append(opportunity)

            # === 针对空头（SELL）持仓的出场策略 ===

            # 1. 价格接近支撑位 - 空头平仓机会
            if support_resistance["nearest_support"] is not None:
                support_price = support_resistance["nearest_support"]
                distance_pct = (current_price - support_price) / current_price * 100

                if distance_pct < 1.5:  # 非常接近支撑位
                    opportunity = {
                        "side": "BUY",  # 平空
                        "price": support_price * 1.005,  # 略高于支撑位
                        "type": "LIMIT",
                        "priority": 2,
                        "reason": f"价格接近支撑位 {support_price:.6f} (距离: {distance_pct:.2f}%)"
                    }
                    opportunities.append(opportunity)

            # 2. 趋势转向或价格预测看涨 - 空头平仓机会
            if trend_shifting_up or direction == "UP":
                reason = "趋势转向上升" if trend_shifting_up else "价格预测看涨"
                if predicted_price and predicted_price > current_price:
                    reason += f", 预测价格上涨至 {predicted_price:.6f}"

                # 如果预测置信度高或趋势转向明显，使用市价单
                if prediction_confidence > 0.7 or trend_shifting_up:
                    opportunity = {
                        "side": "BUY",  # 平空
                        "price": current_price,
                        "type": "MARKET",
                        "priority": 1,
                        "reason": reason
                    }
                else:
                    # 否则使用限价单
                    target_price = current_price * 0.995  # 略低于当前价格
                    opportunity = {
                        "side": "BUY",  # 平空
                        "price": target_price,
                        "type": "LIMIT",
                        "priority": 3,
                        "reason": f"{reason}, 等待轻微下跌至 {target_price:.6f}"
                    }
                opportunities.append(opportunity)

            # 3. 设置获利目标 - 空头平仓机会
            if predicted_price and predicted_price < current_price:
                gain_pct = (current_price - predicted_price) / current_price * 100

                # 针对1-5小时交易，设置适当的获利目标
                if gain_pct >= 3.0:  # 超过3%的获利机会
                    # 在预测价格和当前价格之间设置目标
                    target_price = current_price * (1 - min(gain_pct, 5.0) / 100 * 0.7)  # 取预测收益的70%
                    opportunity = {
                        "side": "BUY",  # 平空
                        "price": target_price,
                        "type": "LIMIT",
                        "priority": 2,
                        "reason": f"设置获利目标，预计下跌 {gain_pct:.2f}%，取 70% 收益"
                    }
                    opportunities.append(opportunity)

            # 对机会进行排序 - 按优先级
            opportunities.sort(key=lambda x: x.get("priority", 999))

            # 更新结果
            result["opportunities"] = opportunities
            if opportunities:
                result["best_opportunity"] = opportunities[0]

            return result

        except Exception as e:
            print_colored(f"❌ 计算出场点失败: {e}", Colors.ERROR)
            return result

    def check_pending_entries(self, symbol: str) -> Dict[str, Any]:
        """检查等待中的入场机会是否可以执行

        参数:
            symbol: 交易对

        返回:
            入场检查结果
        """
        # 初始化结果
        result = {
            "should_enter": False,
            "signal": "NEUTRAL",
            "reason": "",
            "quality_score": 0
        }

        # 检查是否有等待中的入场机会
        if symbol not in self.entry_opportunities:
            return result

        entry_opp = self.entry_opportunities[symbol]

        # 检查是否已过期
        current_time = time.time()
        if current_time > entry_opp.get("expiry_time", 0):
            # 机会已过期
            print_colored(f"⏱️ {symbol}入场机会已过期", Colors.WARNING)
            # 移除过期机会
            self.entry_opportunities.pop(symbol)
            return result

        # 获取当前价格
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # 检查价格是否达到目标
            target_price = entry_opp.get("target_price")
            if target_price:
                signal = entry_opp.get("original_signal", "NEUTRAL")

                # 检查是否达到买入目标价格
                if "BUY" in signal and current_price <= target_price * 1.002:  # 允许0.2%误差
                    result["should_enter"] = True
                    result["signal"] = signal
                    result["reason"] = f"价格已回调至目标价格附近: {current_price:.6f} ≈ {target_price:.6f}"
                    result["quality_score"] = entry_opp.get("quality_score", 0)

                    # 移除已执行的入场机会
                    self.entry_opportunities.pop(symbol)

                    print_colored(f"✅ {symbol}入场机会已触发，可以执行{signal}订单", Colors.GREEN)
                    return result

                # 检查是否达到卖出目标价格
                elif "SELL" in signal and current_price >= target_price * 0.998:  # 允许0.2%误差
                    result["should_enter"] = True
                    result["signal"] = signal
                    result["reason"] = f"价格已反弹至目标价格附近: {current_price:.6f} ≈ {target_price:.6f}"
                    result["quality_score"] = entry_opp.get("quality_score", 0)

                    # 移除已执行的入场机会
                    self.entry_opportunities.pop(symbol)

                    print_colored(f"✅ {symbol}入场机会已触发，可以执行{signal}订单", Colors.GREEN)
                    return result

            # 检查当前价格与目标价格的距离
            if target_price:
                distance_pct = abs(current_price - target_price) / target_price * 100
                print_colored(
                    f"⏳ {symbol}等待入场中 - 当前价格: {current_price:.6f}, "
                    f"目标价格: {target_price:.6f}, 距离: {distance_pct:.2f}%",
                    Colors.CYAN
                )

            return result
        except Exception as e:
            print_colored(f"❌ 检查{symbol}等待入场失败: {e}", Colors.ERROR)
            return result