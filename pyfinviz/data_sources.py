from typing import Any

import pandas as pd
import requests
import yfinance as yf

FMP_V3_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE_URL = "https://financialmodelingprep.com/stable"


def get_yfinance_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval, auto_adjust=True)


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
