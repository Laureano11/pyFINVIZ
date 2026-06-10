from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Rango canónico de descarga: el cache SIEMPRE se baja y guarda completo en este
# rango, independiente del sub-rango que pida cada consulta. Así un backtest
# in-sample (p.ej. 2005-2015) nunca recorta el cache amplio (regla anti-corrupción).
CANONICAL_START = "2000-01-01"
CANONICAL_END = "2030-01-01"


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace(".", "-")
    return CACHE_DIR / f"{safe}.parquet"


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Recorta el cache (que puede tener más historia) al rango pedido."""
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    return df.loc[(df.index >= lo) & (df.index <= hi)]


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_ticker(
    ticker: str,
    start: str,
    end: str,
    *,
    refresh: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """Carga OHLCV ajustado de un ticker, cacheando en parquet.

    El cache se baja y guarda SIEMPRE en el rango canónico amplio; `start`/`end`
    solo recortan lo que se devuelve. Un sub-rango nunca sobreescribe el cache
    amplio.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)

    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if not cached.empty:
            return _slice(cached, start, end)

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = _download_one(ticker, CANONICAL_START, CANONICAL_END)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            df = pd.DataFrame()
        if not df.empty:
            df.to_parquet(path)
            return _slice(df, start, end)
        time.sleep(1.0 + attempt)

    if last_exc is not None:
        print(f"[loader] {ticker}: descarga fallida ({last_exc}).")
    else:
        print(f"[loader] {ticker}: sin datos.")
    return pd.DataFrame()


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    field: str = "Close",
    refresh: bool = False,
) -> pd.DataFrame:
    """Devuelve un DataFrame fechas × tickers con el campo pedido.

    Los tickers sin datos se omiten. El índice es la unión de fechas; cada
    columna conserva sus NaN propios (el motor maneja activos con historia corta).
    """
    series: dict[str, pd.Series] = {}
    for ticker in dict.fromkeys(tickers):  # dedup preservando orden
        df = load_ticker(ticker, start, end, refresh=refresh)
        if df.empty or field not in df.columns:
            continue
        series[ticker] = df[field]

    if not series:
        return pd.DataFrame()

    prices = pd.DataFrame(series).sort_index()
    return prices


def load_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    *,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Devuelve un dict ticker -> OHLCV (para estrategias que necesitan más que Close)."""
    out: dict[str, pd.DataFrame] = {}
    for ticker in dict.fromkeys(tickers):
        df = load_ticker(ticker, start, end, refresh=refresh)
        if not df.empty:
            out[ticker] = df
    return out
