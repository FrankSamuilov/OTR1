import numpy as np
import pandas as pd


class MarketEnvironmentClassifier:
    def __init__(self):
        # 环境分类阈值
        self.volatility_threshold = 1.5  # ATR相对于均值的倍数，超过视为高波动
        self.trend_strength_threshold = 25  # ADX阈值，超过视为强趋势
        self.breakout_threshold = 2.0  # 价格突破布林带的倍数，超过视为突破
        self.momentum_threshold = 0.8  # 动量指标阈值，用于确认趋势

    def classify_environment(self, df):
        """
        根据市场指标将当前环境分类为: 趋势、震荡、突破或极端波动

        参数:
            df: 包含指标的DataFrame(需要含有'close', 'ATR', 'ADX'等指标)

        返回:
            环境类型及详细信息
        """
        if df is None or len(df) < 20:
            return {
                'environment': 'unknown',
                'confidence': 0,
                'details': '数据不足'
            }

        try:
            # 提取关键指标
            close = df['close'].values
            atr = df['ATR'].values if 'ATR' in df.columns else None
            adx = df['ADX'].values if 'ADX' in df.columns else None
            bb_upper = df['BB_Upper'].values if 'BB_Upper' in df.columns else None
            bb_lower = df['BB_Lower'].values if 'BB_Lower' in df.columns else None
            bb_middle = df['BB_Middle'].values if 'BB_Middle' in df.columns else None

            # 创建结果对象
            result = {
                'environment': 'unknown',
                'confidence': 0,
                'details': {}
            }

            # 1. 检测极端波动
            if atr is not None:
                atr_mean = np.mean(atr[:-20]) if len(atr) > 20 else np.mean(atr)
                current_atr = atr[-1]
                atr_ratio = current_atr / atr_mean if atr_mean > 0 else 1.0

                if atr_ratio > self.volatility_threshold * 2:  # 非常极端的波动
                    result['environment'] = 'extreme_volatility'
                    result['confidence'] = min(100, atr_ratio * 30)
                    result['details'] = {
                        'atr_ratio': atr_ratio,
                        'current_atr': current_atr,
                        'atr_mean': atr_mean
                    }
                    return result

            # 2. 检测突破
            if bb_upper is not None and bb_lower is not None:
                current_close = close[-1]
                current_upper = bb_upper[-1]
                current_lower = bb_lower[-1]

                # 计算布林带宽度
                bb_width = (
                                       current_upper - current_lower) / current_middle if 'BB_Middle' in df.columns and current_middle > 0 else 0

                # 计算突破程度
                upper_breakout_ratio = (current_close - current_upper) / (
                            current_upper - current_middle) if current_upper > current_middle else 0
                lower_breakout_ratio = (current_lower - current_close) / (
                            current_middle - current_lower) if current_middle > current_lower else 0

                # 检测是否为有效突破
                if upper_breakout_ratio > self.breakout_threshold or lower_breakout_ratio > self.breakout_threshold:
                    result['environment'] = 'breakout'
                    result['confidence'] = min(100, max(upper_breakout_ratio, lower_breakout_ratio) * 40)
                    result['details'] = {
                        'breakout_direction': 'upward' if upper_breakout_ratio > lower_breakout_ratio else 'downward',
                        'breakout_ratio': max(upper_breakout_ratio, lower_breakout_ratio),
                        'bb_width': bb_width
                    }
                    return result

            # 3. 检测趋势或震荡
            if adx is not None:
                current_adx = adx[-1]

                # 使用超级趋势和EMA判断趋势方向
                trend_direction = 'neutral'
                if 'Supertrend_Direction' in df.columns:
                    st_direction = df['Supertrend_Direction'].iloc[-1]
                    trend_direction = 'uptrend' if st_direction > 0 else 'downtrend' if st_direction < 0 else 'neutral'
                elif 'EMA5' in df.columns and 'EMA20' in df.columns:
                    ema_short = df['EMA5'].iloc[-1]
                    ema_long = df['EMA20'].iloc[-1]
                    trend_direction = 'uptrend' if ema_short > ema_long else 'downtrend' if ema_short < ema_long else 'neutral'

                # 检查动量确认
                momentum_confirms = False
                if 'Momentum' in df.columns:
                    momentum = df['Momentum'].iloc[-1]
                    momentum_confirms = (momentum > 0 and trend_direction == 'uptrend') or (
                                momentum < 0 and trend_direction == 'downtrend')

                if current_adx > self.trend_strength_threshold and trend_direction != 'neutral' and momentum_confirms:
                    result['environment'] = 'trending'
                    result['confidence'] = min(100, current_adx * 2)
                    result['details'] = {
                        'adx': current_adx,
                        'trend_direction': trend_direction,
                        'momentum_confirms': momentum_confirms
                    }
                else:
                    result['environment'] = 'ranging'
                    result['confidence'] = min(100, (
                                self.trend_strength_threshold - current_adx) * 3) if current_adx < self.trend_strength_threshold else 20
                    result['details'] = {
                        'adx': current_adx,
                        'bb_width': bb_width if 'bb_width' in locals() else None
                    }

                return result

            # 如果没有足够的指标，使用简化判断
            if 'bb_width' in locals():
                if bb_width < 0.05:  # 紧缩的布林带，可能是区间震荡
                    result['environment'] = 'ranging'
                    result['confidence'] = 60
                    result['details'] = {'bb_width': bb_width}
                else:
                    result['environment'] = 'unknown'
                    result['confidence'] = 30
                    result['details'] = {'reason': '指标不足，无法准确判断'}

            return result

        except Exception as e:
            return {
                'environment': 'error',
                'confidence': 0,
                'details': str(e)
            }

    def get_optimal_strategy_params(self, environment_info):
        """
        根据市场环境获取最优策略参数

        参数:
            environment_info: 市场环境信息

        返回:
            优化后的策略参数
        """
        env_type = environment_info['environment']
        confidence = environment_info['confidence']

        # 默认参数
        params = {
            'entry_threshold': 7.0,  # 入场质量评分阈值
            'position_size_pct': 5.0,  # 仓位大小（账户百分比）
            'stop_loss_pct': 3.0,  # 止损百分比
            'take_profit_pct': 6.0,  # 止盈百分比
            'trailing_stop': False,  # 是否启用移动止损
            'use_hedge': False,  # 是否使用对冲
            'max_leverage': 5  # 最大杠杆
        }

        # 根据环境类型调整参数
        if env_type == 'trending':
            # 趋势市场：更激进的仓位，更宽松的止损，更高的止盈
            direction = environment_info['details'].get('trend_direction', 'neutral')

            params['entry_threshold'] = 6.0  # 降低入场门槛
            params['position_size_pct'] = 8.0  # 增加仓位
            params['stop_loss_pct'] = 4.0  # 宽松止损
            params['take_profit_pct'] = 12.0  # 更高止盈
            params['trailing_stop'] = True  # 启用移动止损
            params['preferred_direction'] = direction  # 优先方向
            params['max_leverage'] = 10  # 更高杠杆

        elif env_type == 'ranging':
            # 震荡市场：更保守的策略，小仓位，严格止损止盈
            params['entry_threshold'] = 8.0  # 提高入场门槛
            params['position_size_pct'] = 3.0  # 减小仓位
            params['stop_loss_pct'] = 2.0  # 严格止损
            params['take_profit_pct'] = 4.0  # 更低止盈目标
            params['max_leverage'] = 3  # 更低杠杆

        elif env_type == 'breakout':
            # 突破市场：中等仓位，关注突破方向
            direction = environment_info['details'].get('breakout_direction', 'unknown')

            params['entry_threshold'] = 7.0  # 标准入场门槛
            params['position_size_pct'] = 5.0  # 标准仓位
            params['stop_loss_pct'] = 3.5  # 适中止损
            params['take_profit_pct'] = 7.0  # 适中止盈
            params['preferred_direction'] = 'upward' if direction == 'upward' else 'downward'
            params['max_leverage'] = 7  # 中等杠杆

        elif env_type == 'extreme_volatility':
            # 极端波动：保守策略，非常小的仓位，或完全避免交易
            params['entry_threshold'] = 9.0  # 非常高的入场门槛
            params['position_size_pct'] = 2.0  # 非常小的仓位
            params['stop_loss_pct'] = 5.0  # 更宽的止损以适应波动
            params['take_profit_pct'] = 10.0  # 更高的止盈目标
            params['use_hedge'] = True  # 使用对冲策略
            params['max_leverage'] = 2  # 最低杠杆

        # 根据置信度调整参数，低置信度时更保守
        if confidence < 70:
            # 降低仓位
            params['position_size_pct'] = max(2.0, params['position_size_pct'] * 0.7)
            # 提高入场门槛
            params['entry_threshold'] = min(9.0, params['entry_threshold'] * 1.2)
            # 降低杠杆
            params['max_leverage'] = max(2, round(params['max_leverage'] * 0.7))

        return params