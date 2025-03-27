import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta




class PerformanceMonitor:
    def __init__(self, data_file='performance_data.json', log_dir='logs'):
        self.data_file = data_file
        self.log_dir = log_dir
        self.performance_data = self.load_data()

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 性能指标
        self.metrics = {
            'total_trades': 0,  # 总交易次数
            'winning_trades': 0,  # 盈利交易次数
            'losing_trades': 0,  # 亏损交易次数
            'total_profit': 0.0,  # 总利润
            'total_loss': 0.0,  # 总亏损
            'max_drawdown': 0.0,  # 最大回撤
            'max_drawdown_period': [],  # 最大回撤期间
            'daily_returns': {},  # 每日收益率
            'balance_history': [],  # 余额历史
            'trade_history': [],  # 交易历史
            'symbol_performance': {}  # 各交易对表现
}

    def load_data(self):
        """加载历史性能数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"加载性能数据失败: {e}")
            return {}

    def save_data(self):
        """保存性能数据"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=4)
        except Exception as e:
            print(f"保存性能数据失败: {e}")

    def record_trade(self, trade_info):
        """
        记录一笔交易

        参数:
            trade_info: 包含交易详情的字典:
                {
                    'symbol': 交易对,
                    'side': 方向 (BUY/SELL),
                    'entry_price': 入场价格,
                    'exit_price': 出场价格,
                    'quantity': 数量,
                    'profit_loss': 盈亏金额,
                    'profit_percentage': 盈亏百分比,
                    'entry_time': 入场时间,
                    'exit_time': 出场时间,
                    'trade_duration': 持仓时间(分钟),
                    'reason': 出场原因
                }
        """
        # 更新交易历史
        self.metrics['trade_history'].append(trade_info)

        # 更新总交易次数
        self.metrics['total_trades'] += 1

        # 更新盈亏统计
        if trade_info['profit_loss'] > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['total_profit'] += trade_info['profit_loss']
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['total_loss'] += abs(trade_info['profit_loss'])

        # 更新交易对统计
        symbol = trade_info['symbol']
        if symbol not in self.metrics['symbol_performance']:
            self.metrics['symbol_performance'][symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'average_profit': 0.0,
                'win_rate': 0.0
            }

        self.metrics['symbol_performance'][symbol]['total_trades'] += 1
        if trade_info['profit_loss'] > 0:
            self.metrics['symbol_performance'][symbol]['winning_trades'] += 1
            self.metrics['symbol_performance'][symbol]['total_profit'] += trade_info['profit_loss']
        else:
            self.metrics['symbol_performance'][symbol]['total_loss'] += abs(trade_info['profit_loss'])

        # 更新胜率和平均盈利
        symbol_data = self.metrics['symbol_performance'][symbol]
        symbol_data['win_rate'] = symbol_data['winning_trades'] / symbol_data['total_trades'] * 100
        total_pnl = symbol_data['total_profit'] - symbol_data['total_loss']
        symbol_data['average_profit'] = total_pnl / symbol_data['total_trades']

        # 保存更新后的数据
        self.performance_data.update(self.metrics)
        self.save_data()

    def update_balance(self, balance, timestamp=None):
        """
        更新账户余额历史

        参数:
            balance: 当前账户余额
            timestamp: 时间戳，默认为当前时间
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.metrics['balance_history'].append({
            'timestamp': timestamp,
            'balance': balance
        })

        # 计算最大回撤
        balances = [entry['balance'] for entry in self.metrics['balance_history']]

        if len(balances) > 1:
            # 计算累计最大值
            peak_balances = []
            max_balance = balances[0]

            for balance in balances:
                if balance > max_balance:
                    max_balance = balance
                peak_balances.append(max_balance)

            # 计算回撤
            drawdowns = [(peak - balance) / peak for peak, balance in zip(peak_balances, balances)]
            max_drawdown_idx = np.argmax(drawdowns)

            if drawdowns[max_drawdown_idx] > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = drawdowns[max_drawdown_idx]

                # 找到最大回撤区间
                peak_idx = 0
                for i in range(max_drawdown_idx, -1, -1):
                    if balances[i] == peak_balances[max_drawdown_idx]:
                        peak_idx = i
                        break

                self.metrics['max_drawdown_period'] = [
                    self.metrics['balance_history'][peak_idx]['timestamp'],
                    self.metrics['balance_history'][max_drawdown_idx]['timestamp']
                ]

        # 更新每日收益
        today = datetime.now().strftime("%Y-%m-%d")
        if len(self.metrics['balance_history']) > 1:
            prev_day_balance = None

            # 寻找昨日最后余额
            for entry in reversed(self.metrics['balance_history'][:-1]):
                entry_date = entry['timestamp'].split()[0]
                if entry_date != today:
                    prev_day_balance = entry['balance']
                    break

            if prev_day_balance is not None:
                daily_return = (balance - prev_day_balance) / prev_day_balance * 100
                self.metrics['daily_returns'][today] = daily_return

        # 保存更新后的数据
        self.performance_data.update(self.metrics)
        self.save_data()

    def generate_report(self, output_file='performance_report.html'):
        """
        生成性能报告

        参数:
            output_file: 输出文件名
        """
        # 计算关键指标
        win_rate = self.metrics['winning_trades'] / max(1, self.metrics['total_trades']) * 100
        profit_factor = self.metrics['total_profit'] / max(1, self.metrics['total_loss'])
        average_profit = sum(trade['profit_loss'] for trade in self.metrics['trade_history']) / max(1, len(
            self.metrics['trade_history']))
        average_win = sum(
            trade['profit_loss'] for trade in self.metrics['trade_history'] if trade['profit_loss'] > 0) / max(1,
                                                                                                               self.metrics[
                                                                                                                   'winning_trades'])
        average_loss = sum(
            abs(trade['profit_loss']) for trade in self.metrics['trade_history'] if trade['profit_loss'] < 0) / max(1,
                                                                                                                    self.metrics[
                                                                                                                        'losing_trades'])

        # 创建报告
        report = f"""
        <html>
        <head>
            <title>交易系统性能报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>交易系统性能报告</h1>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>关键性能指标</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>总交易次数</td><td>{self.metrics['total_trades']}</td></tr>
                <tr><td>盈利交易次数</td><td>{self.metrics['winning_trades']}</td></tr>
                <tr><td>亏损交易次数</td><td>{self.metrics['losing_trades']}</td></tr>
                <tr><td>胜率</td><td>{win_rate:.2f}%</td></tr>
                <tr><td>盈亏比</td><td>{profit_factor:.2f}</td></tr>
                <tr><td>总利润</td><td class="{'positive' if self.metrics['total_profit'] > self.metrics['total_loss'] else 'negative'}">{self.metrics['total_profit'] - self.metrics['total_loss']:.2f} USDC</td></tr>
                <tr><td>平均交易盈亏</td><td class="{'positive' if average_profit > 0 else 'negative'}">{average_profit:.2f} USDC</td></tr>
                <tr><td>平均盈利</td><td class="positive">{average_win:.2f} USDC</td></tr>
                <tr><td>平均亏损</td><td class="negative">-{average_loss:.2f} USDC</td></tr>
                <tr><td>最大回撤</td><td class="negative">{self.metrics['max_drawdown'] * 100:.2f}%</td></tr>
            </table>

            <h2>交易对性能</h2>
            <table>
                <tr>
                    <th>交易对</th>
                    <th>交易次数</th>
                    <th>胜率</th>
                    <th>平均盈亏</th>
                    <th>总盈亏</th>
                </tr>
        """

        # 添加各交易对性能
        for symbol, data in self.metrics['symbol_performance'].items():
            total_pnl = data['total_profit'] - data['total_loss']
            report += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>{data['total_trades']}</td>
                    <td>{data['win_rate']:.2f}%</td>
                    <td class="{'positive' if data['average_profit'] > 0 else 'negative'}">{data['average_profit']:.2f} USDC</td>
                    <td class="{'positive' if total_pnl > 0 else 'negative'}">{total_pnl:.2f} USDC</td>
                </tr>
            """

        report += """
            </table>

            <h2>最近交易记录</h2>
            <table>
                <tr>
                    <th>交易对</th>
                    <th>方向</th>
                    <th>入场价格</th>
                    <th>出场价格</th>
                    <th>盈亏</th>
                    <th>盈亏比例</th>
                    <th>持仓时间</th>
                    <th>出场原因</th>
                </tr>
        """

        # 添加最近10笔交易记录
        recent_trades = sorted(self.metrics['trade_history'], key=lambda x: x['exit_time'], reverse=True)[:10]

        for trade in recent_trades:
            report += f"""
                <tr>
                    <td>{trade['symbol']}</td>
                    <td>{trade['side']}</td>
                    <td>{trade['entry_price']:.4f}</td>
                    <td>{trade['exit_price']:.4f}</td>
                    <td class="{'positive' if trade['profit_loss'] > 0 else 'negative'}">{trade['profit_loss']:.2f} USDC</td>
                    <td class="{'positive' if trade['profit_percentage'] > 0 else 'negative'}">{trade['profit_percentage']:.2f}%</td>
                    <td>{trade['trade_duration']} 分钟</td>
                    <td>{trade['reason']}</td>
                </tr>
            """

        report += """
            </table>
        </body>
        </html>
        """

        # 保存报告
        with open(output_file, 'w') as f:
            f.write(report)

        return output_file