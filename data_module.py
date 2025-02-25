import pandas as pd

def get_historical_data(client, symbol):
    try:
        candles = client.futures_klines(symbol=symbol, interval="15m", limit=100)
        df = pd.DataFrame(candles, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        return None

def get_spot_balance(client):
    try:
        info = client.get_asset_balance(asset="USDC")
        return float(info["free"])
    except Exception as e:
        return 0.0

def get_futures_balance(client):
    try:
        assets = client.futures_account_balance()
        for asset in assets:
            if asset["asset"] == "USDC":
                return float(asset["balance"])
        return 0.0
    except Exception as e:
        return 0.0
