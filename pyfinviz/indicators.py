import pandas as pd


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA14"] = df["Close"].ewm(span=14, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    fast_ema = df["Close"].ewm(span=12, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = fast_ema - slow_ema
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RSI"] = compute_rsi(df["Close"], 14)
    df["RSI2"] = compute_rsi(df["Close"], 2)

    highest_high_14 = df["High"].rolling(14).max()
    lowest_low_14 = df["Low"].rolling(14).min()
    range_14 = (highest_high_14 - lowest_low_14).replace(0, pd.NA)
    df["WILLR"] = -100 * ((highest_high_14 - df["Close"]) / range_14)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
    positive_mf_14 = positive_flow.rolling(14).sum()
    negative_mf_14 = negative_flow.rolling(14).sum().replace(0, pd.NA)
    money_flow_ratio = positive_mf_14 / negative_mf_14
    df["MFI14"] = 100 - (100 / (1 + money_flow_ratio))

    rsi_min_14 = df["RSI"].rolling(14).min()
    rsi_max_14 = df["RSI"].rolling(14).max()
    rsi_range_14 = (rsi_max_14 - rsi_min_14).replace(0, pd.NA)
    stoch_rsi = (df["RSI"] - rsi_min_14) / rsi_range_14
    df["STOCH_RSI_K"] = stoch_rsi.rolling(3).mean() * 100
    df["STOCH_RSI_D"] = df["STOCH_RSI_K"].rolling(3).mean()

    prev_close = df["Close"].shift(1)
    tr_components = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    df["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    up_move = df["High"].diff()
    down_move = df["Low"].shift(1) - df["Low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_dm_14 = plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    minus_dm_14 = minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atr_safe = df["ATR14"].replace(0, pd.NA)
    df["PLUS_DI14"] = 100 * (plus_dm_14 / atr_safe)
    df["MINUS_DI14"] = 100 * (minus_dm_14 / atr_safe)
    di_sum = (df["PLUS_DI14"] + df["MINUS_DI14"]).replace(0, pd.NA)
    dx = 100 * ((df["PLUS_DI14"] - df["MINUS_DI14"]).abs() / di_sum)
    df["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    daily_range = (df["High"] - df["Low"]).replace(0, pd.NA)
    df["IBS"] = (df["Close"] - df["Low"]) / daily_range

    return df
