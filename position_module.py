def load_positions(client):
    try:
        positions = client.futures_position_information()
        open_positions = []
        for pos in positions:
            amt = float(pos.get("positionAmt", 0))
            if abs(amt) > 0:
                open_positions.append({
                    "symbol": pos["symbol"],
                    "side": "BUY" if amt > 0 else "SELL",
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "quantity": abs(amt),
                    "open_time": float(pos.get("updateTime", 0)) / 1000,
                    "max_profit": 0.0
                })
        return open_positions
    except Exception as e:
        return []
