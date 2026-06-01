from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any
from threading import Lock

import pandas as pd
import requests
import yfinance as yf
import time

from pyfinviz.ibkr import get_ibkr_history

FMP_V3_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_STABLE_BASE_URL = "https://financialmodelingprep.com/stable"
_YFINANCE_LOCK = Lock()


def _clean_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _fetch_yfinance_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    with _YFINANCE_LOCK:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is not None and not df.empty:
            return df

        return yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            threads=False,
            auto_adjust=True,
        )


@lru_cache(maxsize=512)
def _cached_yfinance_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return _fetch_yfinance_history(symbol, period, interval)


def get_yfinance_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    symbol = _clean_symbol(symbol)
    max_attempts = 3
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            df = _cached_yfinance_history(symbol, period, interval)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()

            last_exc = None
        except Exception as e:
            last_exc = e

        time.sleep(1 * attempt)

    return pd.DataFrame()


def _download_chunk(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out

    if not isinstance(data.columns, pd.MultiIndex):
        df = data.dropna(how="all")
        if not df.empty and len(symbols) == 1:
            out[symbols[0]] = df
        return out

    available = set(data.columns.get_level_values(0))
    for symbol in symbols:
        if symbol not in available:
            continue
        df = data[symbol].dropna(how="all")
        if not df.empty:
            out[symbol] = df
    return out


def get_yfinance_history_batch(
    symbols: list[str],
    period: str = "2y",
    interval: str = "1d",
    chunk_size: int = 40,
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    clean = list(dict.fromkeys(_clean_symbol(s) for s in symbols if s and s.strip()))
    if not clean:
        return {}

    chunks = [clean[i : i + chunk_size] for i in range(0, len(clean), chunk_size)]
    workers = max(1, min(max_workers, len(chunks)))

    histories: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_chunk, chunk, period, interval) for chunk in chunks]
        for future in as_completed(futures):
            try:
                histories.update(future.result())
            except Exception:
                continue

    return histories


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
