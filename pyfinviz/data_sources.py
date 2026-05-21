from typing import Any

import pandas as pd
import requests
import yfinance as yf

from pyfinviz.ibkr import get_ibkr_history

FMP_V3_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE_URL = "https://financialmodelingprep.com/stable"


def get_yfinance_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval, auto_adjust=True)


def get_market_history(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    source: str = "yfinance",
    fallback_to_yfinance: bool = False,
    ibkr_host: str = "127.0.0.1",
    ibkr_port: int = 7497,
    ibkr_client_id: int = 1,
    ibkr_use_rth: bool = True,
    ibkr_what_to_show: str = "TRADES",
) -> pd.DataFrame:
    normalized_source = (source or "yfinance").strip().lower()
    if normalized_source == "ibkr":
        try:
            return get_ibkr_history(
                symbol=symbol,
                period=period,
                interval=interval,
                host=ibkr_host,
                port=ibkr_port,
                client_id=ibkr_client_id,
                use_rth=ibkr_use_rth,
                what_to_show=ibkr_what_to_show,
            )
        except Exception as exc:
            if not fallback_to_yfinance:
                raise
            print(f"IBKR fallo para {symbol}: {exc}. Se usa yfinance como fallback.")

    return get_yfinance_history(symbol=symbol, period=period, interval=interval)


def get_fmp_profile(symbol: str, api_key: str) -> dict[str, Any]:
    v3_url = f"{FMP_V3_BASE_URL}/profile/{symbol}"
    v3_response = requests.get(v3_url, params={"apikey": api_key}, timeout=20)
    if v3_response.status_code not in (401, 403):
        v3_response.raise_for_status()
        v3_data = v3_response.json()
        return v3_data[0] if v3_data else {}

    stable_url = f"{FMP_STABLE_BASE_URL}/profile"
    stable_response = requests.get(
        stable_url,
        params={"symbol": symbol, "apikey": api_key},
        timeout=20,
    )
    stable_response.raise_for_status()
    stable_data = stable_response.json()
    return stable_data[0] if stable_data else {}


def get_fmp_quote(symbol: str, api_key: str) -> dict[str, Any]:
    v3_url = f"{FMP_V3_BASE_URL}/quote/{symbol}"
    v3_response = requests.get(v3_url, params={"apikey": api_key}, timeout=20)
    if v3_response.status_code not in (401, 403):
        v3_response.raise_for_status()
        v3_data = v3_response.json()
        return v3_data[0] if v3_data else {}

    stable_url = f"{FMP_STABLE_BASE_URL}/quote"
    stable_response = requests.get(
        stable_url,
        params={"symbol": symbol, "apikey": api_key},
        timeout=20,
    )
    stable_response.raise_for_status()
    stable_data = stable_response.json()
    return stable_data[0] if stable_data else {}
