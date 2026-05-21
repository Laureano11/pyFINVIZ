from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def normalize_ibkr_duration(period: str) -> str:
    value = (period or "").strip().lower()
    if not value:
        return "1 Y"

    if value == "max":
        raise ValueError("IBKR no admite period='max'. Usa un periodo finito como 1y o 10y.")

    suffix_map = {
        "y": "Y",
        "yr": "Y",
        "yrs": "Y",
        "year": "Y",
        "years": "Y",
        "mo": "M",
        "month": "M",
        "months": "M",
        "d": "D",
        "day": "D",
        "days": "D",
        "w": "W",
        "wk": "W",
        "week": "W",
        "weeks": "W",
    }

    numeric_part = ""
    suffix_part = ""
    for char in value:
        if char.isdigit():
            numeric_part += char
        else:
            suffix_part += char

    if not numeric_part:
        raise ValueError(f"No se pudo convertir el periodo '{period}' a duration de IBKR.")

    suffix = suffix_map.get(suffix_part, "")
    if not suffix:
        raise ValueError(f"No se pudo convertir el periodo '{period}' a duration de IBKR.")

    return f"{int(numeric_part)} {suffix}"


def normalize_ibkr_bar_size(interval: str) -> str:
    value = (interval or "").strip().lower()
    mapping = {
        "1m": "1 min",
        "2m": "2 mins",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "60m": "1 hour",
        "90m": "90 mins",
        "1h": "1 hour",
        "1d": "1 day",
        "5d": "5 days",
        "1wk": "1 week",
        "1mo": "1 month",
        "3mo": "3 months",
    }
    if value not in mapping:
        raise ValueError(f"Intervalo '{interval}' no soportado por IBKR.")
    return mapping[value]


def _bars_to_dataframe(bars: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bar in bars:
        date_value = getattr(bar, "date", None)
        if date_value is None:
            continue

        if isinstance(date_value, datetime):
            timestamp = date_value
        else:
            timestamp = pd.to_datetime(date_value).to_pydatetime()

        rows.append(
            {
                "Date": timestamp,
                "Open": float(getattr(bar, "open")),
                "High": float(getattr(bar, "high")),
                "Low": float(getattr(bar, "low")),
                "Close": float(getattr(bar, "close")),
                "Volume": float(getattr(bar, "volume", 0.0)),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_ibkr_history(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    use_rth: bool = True,
    what_to_show: str = "TRADES",
) -> pd.DataFrame:
    try:
        from ib_insync import IB, Stock
    except ImportError as exc:
        raise RuntimeError(
            "Falta la dependencia ib_insync. Instalala para usar la API de IBKR."
        ) from exc

    duration = normalize_ibkr_duration(period)
    bar_size = normalize_ibkr_bar_size(interval)

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True)
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )
    finally:
        if ib.isConnected():
            ib.disconnect()

    return _bars_to_dataframe(bars)
