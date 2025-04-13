import logging
import os

def get_logger():
    """
    初始化日志系统，支持趋势状态和持续时间记录
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 创建日志目录

    log_file = os.path.join(log_dir, "trade_log.txt")

    logger = logging.getLogger("USDCTradeBot")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    from binance.client import Client
    from data_module import get_historical_data
    client = Client("your_api_key", "your_api_secret")
    df = get_historical_data(client, "ETHUSDT")
    print(f"测试数据: {df}")