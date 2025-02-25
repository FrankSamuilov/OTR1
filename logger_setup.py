import logging
import os

def get_logger():
    logger = logging.getLogger("USDCTradeBot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(os.path.join(log_dir, "bot.log"), encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
