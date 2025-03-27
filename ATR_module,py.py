"""
入场条件过滤和入场价格计算模块
用于判断是否满足入场条件并计算入场价格、止损止盈位
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from logger_utils import Colors, print_colored, log_entry_signal


class EntryFilter:
    """入场条件过滤类，用于判断是否满足入场条件"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化入场过滤器

        参数:
            config: 配置参数，包含各种阈值
        """
        # 默认配置
        self.default_config = {
            # 超级趋势相关配置
            "min_supertrend_consensus": 0.8,  # 最小超级趋势共识度
            "min_supertrend_strength": 1.2,  # 最小超级趋势强度

            # 恐慌指数配置
            "min_panic_index": 6.5,  # 最小恐慌指数
            "max_panic_index": 9.0,  # 最大恐慌指数

            # 布林带相关配置
            "require_bb_breakout": True,  # 是否要求布林带突破
            "bb_breakout_threshold": 0.02,  # 布林带突破阈值（相对中轨的百分比）

            # 威廉指标配置
            "williams_oversold": -80,  # 威廉指标超卖阈值
            "williams_overbought": -20,  # 威廉指标超买阈值

            # 质量评分配置
            "min_quality_score": 7.0,  # 最小质量评分

            # ADX配置
            "min_adx": 20.0,  # 最小ADX值（趋势强度）

            # 止损止盈配置
            "atr_stop_loss_multiplier": 2.0,  # ATR止损乘数
            "atr_take_profit_multiplier": 4.0,  # ATR止盈乘数
            "atr_entry_offset_multiplier": 0.5,  # ATR入场价偏移乘数

            # 杠杆配置
            "max_leverage": 20,  # 最大杠杆
            "min_leverage": 3,  # 最小杠杆
            "trend_leverage_multiplier": 1.5,  # 趋势明确时的杠杆乘数
        }

        # 合并用户配置
        self.config = {**self.default_config, **(config or {})}

    def check_entry_conditions(self, df: pd.DataFrame, quality_score: float,
                               trend_info: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        检查是否满足入场条件

        参数:
            df: 包含指标的DataFrame
            quality_score: 质量评分
            trend_info: 趋势信息字典

        返回:
            (满足条件, 方向, 详细信息)
        """
        if df is None or df.empty:
            print_colored("❌ 数据为空，无法检查入场条件", Colors.ERROR)
            return False, "NONE", {"reason": "数据为空"}

        # 获取当前行的指标值
        last_row = df.iloc[-1]

        # 获取趋势方向
        trend = trend_info.get("trend", "NEUTRAL")
        trend_confidence = trend_info.get("confidence", "无")

        # 如果趋势不明确，直接拒绝入场
        if trend == "NEUTRAL" or trend_confidence in ["无", "低"]:
            print_colored("❌ 趋势不明确或置信度低，拒绝入场", Colors.WARNING)
            return False, "NONE", {"reason": "趋势不明确", "trend": trend, "confidence": trend_confidence}

        # 入场方向 - 根据趋势确定
        entry_direction = "BUY" if trend == "UP" else "SELL"

        # 创建条件结果字典，用于记录每个条件的检查结果
        conditions = {}

        # 1. 超级趋势共识度检查
        if 'Supertrend_Consensus' in df.columns:
            st_consensus = float(last_row['Supertrend_Consensus'])
            st_min_required = self.config["min_supertrend_consensus"]
            conditions["supertrend_consensus"] = {
                "value": st_consensus,
                "required": st_min_required,
                "passed": st_consensus >= st_min_required
            }

            if st_consensus < st_min_required:
                print_colored(
                    f"❌ 超级趋势共识度 ({st_consensus:.2f}) 低于要求 ({st_min_required:.2f})",
                    Colors.RED
                )
            else:
                print_colored(
                    f"✅ 超级趋势共识度 ({st_consensus:.2f}) 符合要求 (>= {st_min_required:.2f})",
                    Colors.GREEN
                )
        else:
            conditions["supertrend_consensus"] = {
                "value": 0,
                "required": self.config["min_supertrend_consensus"],
                "passed": False,
                "error": "指标不存在"
            }
            print_colored("❌ 超级趋势共识度指标不存在", Colors.RED)

        # 2. 恐慌指数检查
        if 'Panic_Index' in df.columns:
            panic_index = float(last_row['Panic_Index'])
            min_panic = self.config["min_panic_index"]
            max_panic = self.config["max_panic_index"]

            panic_in_range = min_panic <= panic_index <= max_panic
            conditions["panic_index"] = {
                "value": panic_index,
                "min_required": min_panic,
                "max_required": max_panic,
                "passed": panic_in_range
            }

            if not panic_in_range:
                if panic_index < min_panic:
                    print_colored(
                        f"❌ 恐慌指数 ({panic_index:.2f}) 低于要求 ({min_panic:.2f})",
                        Colors.RED
                    )
                else:
                    print_colored(
                        f"❌ 恐慌指数 ({panic_index:.2f}) 高于最大值 ({max_panic:.2f})",
                        Colors.RED
                    )
            else:
                print_colored(
                    f"✅ 恐慌指数 ({panic_index:.2f}) 符合要求 ({min_panic:.2f} - {max_panic:.2f})",
                    Colors.GREEN
                )
        else:
            conditions["panic_index"] = {
                "value": 0,
                "min_required": self.config["min_panic_index"],
                "max_required": self.config["max_panic_index"],
                "passed": False,
                "error": "指标不存在"
            }
            print_colored("❌ 恐慌指数指标不存在", Colors.RED)

        # 3. 布林带突破检查
        if all(col in df.columns for col in ['close', 'BB_Middle', 'BB_Upper', 'BB_Lower']):
            price = float(last_row['close'])
            bb_middle = float(last_row['BB_Middle'])
            bb_upper = float(last_row['BB_Upper'])
            bb_lower = float(last_row['BB_Lower'])

            # 计算相对于中轨的偏移比例
            middle_to_upper = abs(bb_upper - bb_middle) / bb_middle
            middle_to_lower = abs(bb_middle - bb_lower) / bb_middle

            # 如果不要求布林带突破，则直接通过
            if not self.config["require_bb_breakout"]:
                conditions["bb_breakout"] = {
                    "value": 0,
                    "required": False,
                    "passed": True,
                    "note": "不要求布林带突破"
                }
                print_colored("✅ 布林带突破检查已禁用，跳过检查", Colors.GREEN)
            else:
                # 计算价格相对中轨的偏移比例
                price_offset = (price - bb_middle) / bb_middle
                threshold = self.config["bb_breakout_threshold"]

                # 根据趋势方向检查是否突破相应轨道
                if trend == "UP":
                    breakout = price_offset > threshold
                    breakout_desc = f"向上突破 ({price_offset:.2%} > {threshold:.2%})"
                else:  # DOWN
                    breakout = price_offset < -threshold
                    breakout_desc = f"向下突破 ({price_offset:.2%} < -{threshold:.2%})"

                conditions["bb_breakout"] = {
                    "value": price_offset,
                    "required": threshold,
                    "passed": breakout
                }

                if breakout:
                    print_colored(f"✅ 价格{breakout_desc}，符合布林带突破要求", Colors.GREEN)
                else:
                    print_colored(f"❌ 价格未{breakout_desc}，不符合布林带突破要求", Colors.RED)
        else:
            conditions["bb_breakout"] = {
                "value": 0,
                "required": self.config["bb_breakout_threshold"],
                "passed": False,
                "error": "指标不存在"
            }
            print_colored("❌ 布林带指标不存在", Colors.RED)

        # 4. 威廉指标检查
        if 'Williams_R' in df.columns:
            williams_r = float(last_row['Williams_R'])

            # 根据趋势方向检查威廉指标是否在合适的区域
            if trend == "UP":
                # 上升趋势，检查是否超卖（可能反转向上）
                passed = williams_r <= self.config["williams_oversold"]
                expect = f"超卖 (<= {self.config['williams_oversold']})"
            else:  # DOWN
                # 下降趋势，检查是否超买（可能反转向下）
                passed = williams_r >= self.config["williams_overbought"]
                expect = f"超买 (>= {self.config['williams_overbought']})"

            conditions["williams_r"] = {
                "value": williams_r,
                "passed": passed
            }

            if passed:
                print_colored(f"✅ 威廉指标 ({williams_r:.2f}) 符合要求，处于{expect}", Colors.GREEN)
            else:
                print_colored(f"⚠️ 威廉指标 ({williams_r:.2f}) 不符合{expect}要求", Colors.YELLOW)
        else:
            conditions["williams_r"] = {
                "value": 0,
                "passed": False,
                "error": "指标不存在"
            }
            print_colored("⚠️ 威廉指标不存在", Colors.YELLOW)

        # 5. 质量评分检查
        conditions["quality_score"] = {
            "value": quality_score,
            "required": self.config["min_quality_score"],
            "passed": quality_score >= self.config["min_quality_score"]
        }

        if quality_score >= self.config["min_quality_score"]:
            print_colored(f"✅ 质量评分 ({quality_score:.2f}) 符合要求 (>= {self.config['min_quality_score']})",
                          Colors.GREEN)
        else:
            print_colored(f"❌ 质量评分 ({quality_score:.2f}) 低于要求 ({self.config['min_quality_score']})", Colors.RED)

        # 6. ADX检查
        if 'ADX' in df.columns:
            adx = float(last_row['ADX'])
            conditions["adx"] = {
                "value": adx,
                "required": self.config["min_adx"],
                "passed": adx >= self.config["min_adx"]
            }

            if adx >= self.config["min_adx"]:
                print_colored(f"✅ ADX ({adx:.2f}) 符合要求 (>= {self.config['min_adx']})", Colors.GREEN)
            else:
                print_colored(f"❌ ADX ({adx:.2f}) 低于要求 ({self.config['min_adx']})", Colors.RED)
        else:
            conditions["adx"] = {
                "value": 0,
                "required": self.config["min_adx"],
                "passed": False,
                "error": "指标不存在"
            }
            print_colored("❌ ADX指标不存在", Colors.RED)

        # 检查必要条件是否满足
        # 1. 超级趋势共识必须满足
        # 2. 恐慌指数必须满足
        # 3. 如果要求布林带突破，则必须满足
        # 4. 质量评分必须满足
        # 5. ADX必须满足
        # 注意：威廉指标为辅助条件，不强制要求

        critical_conditions = [
            "supertrend_consensus",
            "panic_index",
            "quality_score",
            "adx"
        ]

        # 如果要求布林带突破，加入关键条件
        if self.config["require_bb_breakout"]:
            critical_conditions.append("bb_breakout")

        # 检查所有关键条件
        all_critical_passed = all(
            conditions.get(cond, {}).get("passed", False)
            for cond in critical_conditions
        )

        # 创建结果字典
        result = {
            "passed": all_critical_passed,
            "direction": entry_direction if all_critical_passed else "NONE",
            "conditions": conditions,
            "trend": trend,
            "confidence": trend_confidence,
            "quality_score": quality_score
        }

        # 总结
        if all_critical_passed:
            print_colored(f"✅ 满足所有入场条件，建议{entry_direction}", Colors.GREEN + Colors.BOLD)
        else:
            print_colored("❌ 不满足所有入场条件，不建议交易", Colors.RED + Colors.BOLD)

        return all_critical_passed, entry_direction if all_critical_passed else "NONE", result

    def calculate_entry_price(self, df: pd.DataFrame, direction: str) -> Dict[str, float]:
        """
        根据ATR计算入场价格、止损位和止盈位

        参数:
            df: 包含指标的DataFrame
            direction: 交易方向 ("BUY" or "SELL")

        返回:
            包含入场价格、止损位和止盈位的字典
        """
        if df is None or df.empty or 'ATR' not in df.columns:
            print_colored("❌ 无法计算入场价格：数据为空或缺少ATR", Colors.ERROR)
            return {
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "risk_reward_ratio": 0
            }

        # 获取当前价格和ATR
        current_price = float(df['close'].iloc[-1])
        atr = float(df['ATR'].iloc[-1])

        # 计算入场价、止损位和止盈位
        if direction == "BUY":
            # 计算入场价（当前价 + ATR偏移）
            entry_price = current_price + (atr * self.config["atr_entry_offset_multiplier"])

            # 计算止损位（入场价 - ATR止损乘数）
            stop_loss = entry_price - (atr * self.config["atr_stop_loss_multiplier"])

            # 计算止盈位（入场价 + ATR止盈乘数）
            take_profit = entry_price + (atr * self.config["atr_take_profit_multiplier"])
        else:  # SELL
            # 计算入场价（当前价 - ATR偏移）
            entry_price = current_price - (atr * self.config["atr_entry_offset_multiplier"])

            # 计算止损位（入场价 + ATR止损乘数）
            stop_loss = entry_price + (atr * self.config["atr_stop_loss_multiplier"])

            # 计算止盈位（入场价 - ATR止盈乘数）
            take_profit = entry_price - (atr * self.config["atr_take_profit_multiplier"])

        # 计算风险回报比
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # 输出结果
        print_colored("\n===== 入场价格计算 =====", Colors.BLUE + Colors.BOLD)
        print_colored(f"当前价格: {current_price:.6f}", Colors.INFO)
        print_colored(f"ATR: {atr:.6f}", Colors.INFO)
        print_colored(
            f"入场价: {entry_price:.6f} (当前价{'加' if direction == 'BUY' else '减'} {atr * self.config['atr_entry_offset_multiplier']:.6f})",
            Colors.INFO)
        print_colored(
            f"止损价: {stop_loss:.6f} (入场价{'减' if direction == 'BUY' else '加'} {atr * self.config['atr_stop_loss_multiplier']:.6f})",
            Colors.RED)
        print_colored(
            f"止盈价: {take_profit:.6f} (入场价{'加' if direction == 'BUY' else '减'} {atr * self.config['atr_take_profit_multiplier']:.6f})",
            Colors.GREEN)
        print_colored(f"风险回报比: {risk_reward_ratio:.2f} (回报 {reward:.6f} / 风险 {risk:.6f})", Colors.BLUE)

        result = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
            "current_price": current_price,
            "atr": atr
        }

        return result

    def calculate_optimal_leverage(self, df: pd.DataFrame, trend_info: Dict[str, Any],
                                   quality_score: float) -> int:
        """
        根据趋势强度和质量评分计算最优杠杆

        参数:
            df: 包含指标的DataFrame
            trend_info: 趋势信息字典
            quality_score: 质量评分

        返回:
            杠杆倍数
        """
        # 默认使用最小杠杆
        leverage = self.config["min_leverage"]

        # 获取趋势置信度
        confidence = trend_info.get("confidence", "无")

        # 获取ADX（趋势强度）
        adx = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0

        # 基于质量评分的基础杠杆
        if quality_score >= 9.0:
            base_leverage = 20
        elif quality_score >= 8.0:
            base_leverage = 15
        elif quality_score >= 7.0:
            base_leverage = 10
        elif quality_score >= 6.0:
            base_leverage = 7
        elif quality_score >= 5.0:
            base_leverage = 5
        else:
            base_leverage = 3

        # 趋势强度调整
        trend_multiplier = 1.0
        if confidence in ["高", "中高"]:
            trend_multiplier = 1.2
        elif confidence == "中":
            trend_multiplier = 1.0
        else:
            trend_multiplier = 0.8

        # ADX调整
        adx_multiplier = 1.0
        if adx >= 30:
            adx_multiplier = 1.2
        elif adx >= 25:
            adx_multiplier = 1.1
        elif adx >= 20:
            adx_multiplier = 1.0
        else:
            adx_multiplier = 0.8

        # 计算最终杠杆
        leverage = int(min(
            self.config["max_leverage"],
            max(self.config["min_leverage"], base_leverage * trend_multiplier * adx_multiplier)
        ))

        print_colored("\n===== 杠杆计算 =====", Colors.BLUE + Colors.BOLD)
        print_colored(f"基础杠杆 (基于质量评分 {quality_score:.2f}): {base_leverage}x", Colors.INFO)
        print_colored(f"趋势调整 (基于置信度 {confidence}): {trend_multiplier:.2f}x", Colors.INFO)
        print_colored(f"ADX调整 (ADX = {adx:.2f}): {adx_multiplier:.2f}x", Colors.INFO)
        print_colored(f"最终杠杆: {leverage}x", Colors.YELLOW + Colors.BOLD)

        return leverage

    def process_entry_decision(self, symbol: str, df: pd.DataFrame, quality_score: float,
                               trend_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整处理入场决策，包括条件检查、价格计算和杠杆计算

        参数:
            symbol: 交易对符号
            df: 包含指标的DataFrame
            quality_score: 质量评分
            trend_info: 趋势信息字典

        返回:
            包含完整入场决策信息的字典
        """
        # 检查入场条件
        should_enter, direction, conditions_result = self.check_entry_conditions(
            df, quality_score, trend_info
        )

        # 如果不应该入场，直接返回
        if not should_enter:
            return {
                "should_enter": False,
                "direction": "NONE",
                "conditions": conditions_result,
                "symbol": symbol
            }

        # 计算入场价格
        price_info = self.calculate_entry_price(df, direction)

        # 计算最优杠杆
        leverage = self.calculate_optimal_leverage(df, trend_info, quality_score)

        # 创建完整结果
        result = {
            "should_enter": True,
            "direction": direction,
            "conditions": conditions_result,
            "prices": price_info,
            "leverage": leverage,
            "symbol": symbol,
            "quality_score": quality_score
        }

        # 打印入场信号摘要
        log_entry_signal(
            symbol=symbol,
            direction=direction,
            quality_score=quality_score,
            entry_price=price_info["entry_price"],
            stop_loss=price_info["stop_loss"],
            take_profit=price_info["take_profit"],
            risk_reward_ratio=price_info["risk_reward_ratio"]
        )

        return result