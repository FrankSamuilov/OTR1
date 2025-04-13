import pandas as pd
from indicators_module import calculate_optimized_indicators
from config import CONFIG

class Backtester:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.config = Config()
        self.signals = []

    def calculate_indicators(self):
        """计算所有配置的指标"""
        self.data = calculate_optimized_indicators(self.data)
        # 检查是否计算成功
        if self.data.empty:
            raise ValueError("指标计算失败，请检查数据格式！")

    def generate_signals(self):
        """生成交易信号（示例策略：威廉超卖 + 超级趋势看涨）"""
        for i in range(1, len(self.data)):
            # 威廉指标超卖且超级趋势看涨
            if (self.data['Williams_R'].iloc[i] <= -80 and
                    self.data['Supertrend_Direction'].iloc[i] == 1):
                self.signals.append(("BUY", self.data.index[i]))
            # 威廉指标超买且超级趋势看跌
            elif (self.data['Williams_R'].iloc[i] >= -20 and
                  self.data['Supertrend_Direction'].iloc[i] == -1):
                self.signals.append(("SELL", self.data.index[i]))

    def run_backtest(self):
        """执行回测"""
        self.calculate_indicators()
        self.generate_signals()

        capital = self.config.BACKTEST["initial_capital"]
        position = 0  # 持仓数量

        for signal, idx in self.signals:
            price = self.data.loc[idx, 'close']
            if signal == "BUY":
                # 计算可买数量（根据风险比例）
                risk_amount = capital * self.config.BACKTEST["risk_ratio"]
                size = risk_amount // price
                if size > 0:
                    capital -= size * price * (1 + self.config.BACKTEST["commission"])
                    position += size
            elif signal == "SELL" and position > 0:
                capital += position * price * (1 - self.config.BACKTEST["commission"])
                position = 0

        # 最终净值
        final_value = capital + position * self.data['close'].iloc[-1]
        print(f"回测结果 | 初始资金: {self.config.BACKTEST['initial_capital']}, 最终净值: {final_value:.2f}")


# 使用示例
if __name__ == "__main__":
    backtester = Backtester("historical_data.csv")
    backtester.run_backtest()