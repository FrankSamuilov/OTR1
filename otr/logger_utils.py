"""
日志工具模块：提供彩色日志输出和格式化功能
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime


# ANSI颜色代码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    # 组合样式
    INFO = BLUE
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    CRITICAL = BOLD + BG_RED

    # 特定指标颜色
    TREND_UP = GREEN
    TREND_DOWN = RED
    TREND_NEUTRAL = GRAY

    OVERSOLD = GREEN + BOLD
    OVERBOUGHT = RED + BOLD


def format_log(message: str, style: str = "") -> str:
    """格式化日志消息，添加颜色样式"""
    return f"{style}{message}{Colors.RESET}"


def print_colored(message: str, style: str = "", timestamp: bool = False) -> None:
    """打印彩色消息，可选时间戳"""
    if timestamp:
        time_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{time_str}] {style}{message}{Colors.RESET}")
    else:
        print(f"{style}{message}{Colors.RESET}")


def log_indicator(symbol: str, name: str, value: float, threshold_up: Optional[float] = None,
                  threshold_down: Optional[float] = None) -> None:
    """
    打印指标值，根据阈值自动着色

    参数:
        symbol: 交易对符号
        name: 指标名称
        value: 指标值
        threshold_up: 上阈值（高于此值显示为超买/看涨）
        threshold_down: 下阈值（低于此值显示为超卖/看跌）
    """
    # 选择颜色
    if threshold_up is not None and value > threshold_up:
        color = Colors.OVERBOUGHT if "RSI" in name or "Williams" in name else Colors.TREND_UP
    elif threshold_down is not None and value < threshold_down:
        color = Colors.OVERSOLD if "RSI" in name or "Williams" in name else Colors.TREND_DOWN
    else:
        color = Colors.RESET

    # 格式化数值（根据指标类型选择精度）
    if isinstance(value, int) or value.is_integer():
        formatted_value = f"{int(value)}"
    elif abs(value) < 0.01:
        formatted_value = f"{value:.6f}"
    elif abs(value) < 1:
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = f"{value:.2f}"

    print(f"📊 {symbol} - {name}: {color}{formatted_value}{Colors.RESET}")


def log_trend(symbol: str, trend: str, confidence: str, duration_minutes: int) -> None:
    """
    打印趋势信息，自动着色

    参数:
        symbol: 交易对符号
        trend: 趋势方向 ("UP", "DOWN", "NEUTRAL")
        confidence: 置信度描述
        duration_minutes: 趋势持续时间（分钟）
    """
    if trend == "UP":
        color = Colors.TREND_UP
        trend_icon = "⬆️"
        trend_text = "上升"
    elif trend == "DOWN":
        color = Colors.TREND_DOWN
        trend_icon = "⬇️"
        trend_text = "下降"
    else:
        color = Colors.TREND_NEUTRAL
        trend_icon = "➡️"
        trend_text = "中性"

    # 计算持续时间的小时表示
    hours = duration_minutes / 60
    duration_text = f"{duration_minutes}分钟" if hours < 1 else f"{hours:.1f}小时"

    # 置信度颜色
    confidence_color = (Colors.GREEN if confidence == "高" else
                        Colors.YELLOW if confidence == "中" else
                        Colors.RED)

    print(f"{trend_icon} {symbol} - 趋势: {color}{trend_text}{Colors.RESET}, "
          f"置信度: {confidence_color}{confidence}{Colors.RESET}, "
          f"持续时间: {duration_text}")


def log_entry_signal(symbol: str, direction: str, quality_score: float,
                     entry_price: float, stop_loss: float, take_profit: float,
                     risk_reward_ratio: float) -> None:
    """
    打印入场信号信息

    参数:
        symbol: 交易对符号
        direction: 交易方向 ("BUY", "SELL")
        quality_score: 质量评分
        entry_price: 入场价格
        stop_loss: 止损价格
        take_profit: 止盈价格
        risk_reward_ratio: 风险回报比
    """
    direction_color = Colors.TREND_UP if direction == "BUY" else Colors.TREND_DOWN
    direction_text = "做多" if direction == "BUY" else "做空"

    # 质量评分颜色
    if quality_score >= 8:
        score_color = Colors.GREEN + Colors.BOLD
    elif quality_score >= 6:
        score_color = Colors.GREEN
    elif quality_score >= 4:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    # 风险回报比颜色
    if risk_reward_ratio >= 3:
        rr_color = Colors.GREEN + Colors.BOLD
    elif risk_reward_ratio >= 2:
        rr_color = Colors.GREEN
    elif risk_reward_ratio >= 1:
        rr_color = Colors.YELLOW
    else:
        rr_color = Colors.RED

    print(f"\n{Colors.BOLD}🎯 {symbol} 入场信号{Colors.RESET}")
    print(f"方向: {direction_color}{direction_text}{Colors.RESET}")
    print(f"质量评分: {score_color}{quality_score:.2f}/10{Colors.RESET}")
    print(f"入场价: {direction_color}{entry_price:.6f}{Colors.RESET}")
    print(f"止损价: {Colors.RED}{stop_loss:.6f}{Colors.RESET}")
    print(f"止盈价: {Colors.GREEN}{take_profit:.6f}{Colors.RESET}")
    print(f"风险回报比: {rr_color}{risk_reward_ratio:.2f}{Colors.RESET}")


def log_market_conditions(symbol: str, btc_change: Optional[float] = None,
                          sentiment: Optional[str] = None,
                          panic_index: Optional[float] = None) -> None:
    """
    打印市场情绪和环境信息

    参数:
        symbol: 交易对符号
        btc_change: BTC价格变化百分比
        sentiment: 市场情绪描述
        panic_index: 恐慌指数(0-10)
    """
    # BTC变化
    if btc_change is not None:
        btc_color = (Colors.GREEN if btc_change > 0 else
                     Colors.RED if btc_change < 0 else
                     Colors.RESET)
        print(f"📈 BTC变化率: {btc_color}{btc_change:.2f}%{Colors.RESET}")

    # 市场情绪
    if sentiment is not None:
        if "看多" in sentiment:
            sentiment_color = Colors.GREEN
        elif "看空" in sentiment:
            sentiment_color = Colors.RED
        else:
            sentiment_color = Colors.RESET

        print(f"🌐 市场情绪: {sentiment_color}{sentiment}{Colors.RESET}")

    # 恐慌指数
    if panic_index is not None:
        if panic_index > 7:
            panic_color = Colors.RED + Colors.BOLD
            panic_desc = "极度恐慌"
        elif panic_index > 5:
            panic_color = Colors.RED
            panic_desc = "恐慌"
        elif panic_index > 3:
            panic_color = Colors.YELLOW
            panic_desc = "谨慎"
        else:
            panic_color = Colors.GREEN
            panic_desc = "平静"

        print(f"😱 恐慌指数: {panic_color}{panic_index:.2f}/10 ({panic_desc}){Colors.RESET}")