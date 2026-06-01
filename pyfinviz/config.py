import os
from dataclasses import dataclass
from typing import Optional

from pyfinviz.strategies import STRATEGIES
from pyfinviz.sp500_list import SP500_TICKERS


@dataclass
class RuntimeConfig:
    symbol: str
    backtest_symbol: str
    backtest_symbols: list[str]
    period: str
    interval: str
    horizon_days: int
    ma200_filter_mode: str
    trade_capital_usd: float
    strategy_key: str
    strategy_params: dict[str, float]
    batch_workers: int
    fmp_api_key: str
    enable_fmp: bool


def parse_bool_env(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_ma200_filter(value: Optional[str]) -> str:
    mode = (value or "2").strip().upper()
    if mode == "0":
        return "NONE"
    if mode == "1":
        return "ABOVE"
    if mode == "2":
        return "BELOW"
    if mode == "UNDER":
        return "BELOW"
    if mode not in {"BELOW", "ABOVE", "NONE"}:
        return "BELOW"
    return mode


def parse_symbols_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [s.strip().upper() for s in value.split(",") if s.strip()]



def parse_backtest_interval(value: Optional[str]) -> str:
    interval = (value or "1d").strip().lower()
    allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
    if interval not in allowed:
        return "1d"
    return interval


def choose_strategy() -> str:
    env_choice = os.getenv("STRATEGY_CHOICE", "").strip()
    if env_choice in STRATEGIES:
        return env_choice

    print("\nSelecciona estrategia para el analisis:")
    for key, strategy in STRATEGIES.items():
        print(f"{key} - {strategy.name} ({strategy.description})")

    try:
        selected = input("Elegi opcion (default 1): ").strip() or "1"
    except EOFError:
        selected = "1"

    if selected not in STRATEGIES:
        print("Opcion invalida. Se usa 1 - Golden Cross.")
        return "1"

    return selected


def load_runtime_config(ask_strategy: bool = True) -> RuntimeConfig:
    if ask_strategy:
        strategy_key = choose_strategy()
    else:
        strategy_key = os.getenv("STRATEGY_CHOICE", "1").strip()
        if strategy_key not in STRATEGIES:
            strategy_key = "1"

    workers_raw = os.getenv("BATCH_WORKERS", "8").strip()
    try:
        batch_workers = max(1, int(workers_raw))
    except ValueError:
        batch_workers = 8

    strategy_params = {
        "rsi_max": float(os.getenv("MACD_RSI_MAX", "40")),
        "rsi_min_sell": float(os.getenv("MACD_RSI_SELL_MIN", "60")),
        "willr_buy_level": float(os.getenv("WILLR_BUY_LEVEL", "-80")),
        "willr_sell_level": float(os.getenv("WILLR_SELL_LEVEL", "-20")),
        "adx_ema_min_adx": float(os.getenv("ADX_EMA_MIN_ADX", "15")),
    }

    env_symbols = parse_symbols_list(os.getenv("BACKTEST_SYMBOLS", ""))
    backtest_symbols = env_symbols if env_symbols else list(SP500_TICKERS)

    return RuntimeConfig(
        symbol=os.getenv("SYMBOL", "AAPL"),
        backtest_symbol=os.getenv("BACKTEST_SYMBOL", "AAPL"),
        backtest_symbols=backtest_symbols,
        period=os.getenv("BACKTEST_PERIOD", "10y"),
        interval=parse_backtest_interval(os.getenv("BACKTEST_INTERVAL", "1d")),
        horizon_days=int(os.getenv("ROI_HORIZON_DAYS", "20")),
        ma200_filter_mode=parse_ma200_filter(os.getenv("MA200_FILTER", "2")),
        trade_capital_usd=float(os.getenv("TRADE_CAPITAL_USD", "1000")),
        strategy_key=strategy_key,
        strategy_params=strategy_params,
        batch_workers=batch_workers,
        fmp_api_key=os.getenv("FMP_API_KEY", "").strip(),
        enable_fmp=parse_bool_env(os.getenv("ENABLE_FMP", "0"), default=False),
    )
