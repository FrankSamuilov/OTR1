import pandas as pd

def get_historical_data(client, symbol):
    """
    获取历史K线数据，确保包含高低点，支持趋势分析
    """
    try:
        print(f"尝试获取 {symbol} 数据...")
        candles = client.futures_klines(symbol=symbol, interval="30m", limit=200)
        if not candles or not isinstance(candles, list):
            print(f"错误：获取 {symbol} 数据失败，返回空或无效列表")
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_asset_volume', 'trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        df = pd.DataFrame(candles, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        close_sum = df['close'].sum()
        if df.empty or close_sum == 0:
            print(f"错误：{symbol} 数据为空或无效，close 列和: {close_sum}")
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'],
                                data=[[pd.Timestamp('1970-01-01')] + [0.0] * 5])
        print(f"{symbol} 获取数据成功，close 列和: {close_sum}")
        return df
    except Exception as e:
        print(f"错误：获取 {symbol} 数据失败 - {e}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'],
                            data=[[pd.Timestamp('1970-01-01')] + [0.0] * 5])

def get_spot_balance(client):
    try:
        info = client.get_asset_balance(asset="USDC")
        return float(info["free"])
    except Exception as e:
        print(f"错误：获取现货余额失败 - {e}")
        return 0.0

def get_futures_balance(client):
    try:
        assets = client.futures_account_balance()
        for asset in assets:
            if asset["asset"] == "USDC":
                return float(asset["balance"])
        return 0.0
    except Exception as e:
        print(f"错误：获取期货余额失败 - {e}")
        return 0.0