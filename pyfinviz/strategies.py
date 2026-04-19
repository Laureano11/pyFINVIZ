from dataclasses import dataclass
from typing import Callable

import pandas as pd


EntryFn = Callable[[pd.DataFrame, str, dict[str, float]], pd.Series]
ExitFn = Callable[[pd.DataFrame, dict[str, float]], pd.Series]


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    description: str
    entry_fn: EntryFn
    exit_fn: ExitFn | None = None
    execution: str = "next_open"
    stop_atr: float = 0.0
    take_atr: float = 0.0


def apply_ma200_filter(signal: pd.Series, df: pd.DataFrame, ma200_filter_mode: str) -> pd.Series:
    if ma200_filter_mode == "ABOVE":
        return signal & (df["Close"] > df["MA200"])
    if ma200_filter_mode == "BELOW":
        return signal & (df["Close"] < df["MA200"])
    return signal


def strategy_golden_cross(df: pd.DataFrame, ma200_filter_mode: str, _: dict[str, float]) -> pd.Series:
    cross_up = (df["MA20"] > df["MA50"]) & (df["MA20"].shift(1) <= df["MA50"].shift(1))
    return apply_ma200_filter(cross_up, df, ma200_filter_mode)


def strategy_macd_rsi(df: pd.DataFrame, ma200_filter_mode: str, params: dict[str, float]) -> pd.Series:
    rsi_max = params.get("rsi_max", 40.0)
    macd_cross_up = (df["MACD"] > df["MACD_SIGNAL"]) & (df["MACD"].shift(1) <= df["MACD_SIGNAL"].shift(1))
    entry = macd_cross_up & (df["RSI"] < rsi_max)
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_macd_rsi_exit(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    rsi_min_sell = params.get("rsi_min_sell", 60.0)
    macd_cross_down = (df["MACD"] < df["MACD_SIGNAL"]) & (df["MACD"].shift(1) >= df["MACD_SIGNAL"].shift(1))
    return macd_cross_down & (df["RSI"] > rsi_min_sell)


def strategy_williams_r(df: pd.DataFrame, ma200_filter_mode: str, params: dict[str, float]) -> pd.Series:
    buy_level = params.get("willr_buy_level", -80.0)
    entry = (df["WILLR"] < buy_level) & (df["WILLR"].shift(1) >= buy_level)
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_williams_r_exit(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    sell_level = params.get("willr_sell_level", -20.0)
    return (df["WILLR"] > sell_level) & (df["WILLR"].shift(1) <= sell_level)


def strategy_ibs(df: pd.DataFrame, ma200_filter_mode: str, params: dict[str, float]) -> pd.Series:
    buy_level = params.get("ibs_buy_level", 0.20)
    entry = df["IBS"] < buy_level
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_ibs_exit(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    sell_level = params.get("ibs_sell_level", 0.80)
    return df["IBS"] > sell_level


def strategy_ibs_rsi2(df: pd.DataFrame, _: str, params: dict[str, float]) -> pd.Series:
    ibs_level = params.get("ibs_buy_level", 0.20)
    rsi2_level = params.get("rsi2_buy_level", 10.0)
    return (df["Close"] > df["MA200"]) & (df["RSI2"] < rsi2_level) & (df["IBS"] < ibs_level)


def strategy_ibs_rsi2_exit(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    rsi2_sell = params.get("rsi2_sell_level", 60.0)
    ibs_sell = params.get("ibs_sell_level", 0.80)
    return (df["RSI2"] > rsi2_sell) | (df["IBS"] > ibs_sell)


def strategy_macd_rsi_mfi(df: pd.DataFrame, ma200_filter_mode: str, params: dict[str, float]) -> pd.Series:
    rsi_max = params.get("rsi_max", 40.0)
    mfi_max = params.get("mfi_max", 40.0)
    macd_cross_up = (df["MACD"] > df["MACD_SIGNAL"]) & (df["MACD"].shift(1) <= df["MACD_SIGNAL"].shift(1))
    entry = macd_cross_up & (df["RSI"] < rsi_max) & (df["MFI14"] < mfi_max)
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_macd_rsi_mfi_exit(df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    rsi_min_sell = params.get("rsi_min_sell", 60.0)
    mfi_min_sell = params.get("mfi_min_sell", 60.0)
    macd_cross_down = (df["MACD"] < df["MACD_SIGNAL"]) & (df["MACD"].shift(1) >= df["MACD_SIGNAL"].shift(1))
    return macd_cross_down & (df["RSI"] > rsi_min_sell) & (df["MFI14"] > mfi_min_sell)


def strategy_stochrsi_macd(df: pd.DataFrame, ma200_filter_mode: str, params: dict[str, float]) -> pd.Series:
    # Essential version: keep only momentum direction + stoch oversold zone.
    stoch_k_max = params.get("stochrsi_k_max", 30.0)
    entry = (df["STOCH_RSI_K"] < stoch_k_max) & (df["MACD"] > df["MACD_SIGNAL"])
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_stochrsi_macd_exit(df: pd.DataFrame, _: dict[str, float]) -> pd.Series:
    return df["MACD"] < df["MACD_SIGNAL"]


def strategy_ema50_200_adx_rsi(df: pd.DataFrame, ma200_filter_mode: str, _: dict[str, float]) -> pd.Series:
    golden_cross = (df["EMA50"] > df["EMA200"]) & (df["EMA50"].shift(1) <= df["EMA200"].shift(1))
    trend_filter = df["ADX14"] > 20
    momentum_filter = (df["RSI"] > 30) & (df["RSI"] < 70)
    entry = golden_cross & trend_filter & momentum_filter
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_ema50_200_adx_rsi_exit(df: pd.DataFrame, _: dict[str, float]) -> pd.Series:
    return (df["EMA50"] < df["EMA200"]) & (df["EMA50"].shift(1) >= df["EMA200"].shift(1))


def strategy_adx_ema_direction(df: pd.DataFrame, ma200_filter_mode: str, params: dict[str, float]) -> pd.Series:
    # Keep only core constraints: trend strength + price above EMA14.
    adx_min = params.get("adx_ema_min_adx", 15.0)
    entry = (df["ADX14"] > adx_min) & (df["Close"] > df["EMA14"])
    return apply_ma200_filter(entry, df, ma200_filter_mode)


def strategy_adx_ema_direction_exit(df: pd.DataFrame, _: dict[str, float]) -> pd.Series:
    # Exit on loss of direction only.
    return df["Close"] < df["EMA14"]


STRATEGIES: dict[str, StrategyDefinition] = {
    "1": StrategyDefinition(
        name="Golden Cross",
        description="MA20 cruza por encima de MA50",
        entry_fn=strategy_golden_cross,
    ),
    "2": StrategyDefinition(
        name="MACD + RSI",
        description="Compra: MACD alcista + RSI<40 | Venta: MACD bajista + RSI>60",
        entry_fn=strategy_macd_rsi,
        exit_fn=strategy_macd_rsi_exit,
    ),
    "3": StrategyDefinition(
        name="Williams %R",
        description="Compra: cae bajo -80 | Venta: sube sobre -20",
        entry_fn=strategy_williams_r,
        exit_fn=strategy_williams_r_exit,
    ),
    "4": StrategyDefinition(
        name="IBS",
        description="Compra IBS < 0.20 | Venta IBS > 0.80",
        entry_fn=strategy_ibs,
        exit_fn=strategy_ibs_exit,
        execution="close",
    ),
    "5": StrategyDefinition(
        name="IBS + RSI(2)",
        description="Compra: Close>MA200 + RSI2<10 + IBS<0.20 | Venta: RSI2>60 o IBS>0.80",
        entry_fn=strategy_ibs_rsi2,
        exit_fn=strategy_ibs_rsi2_exit,
        execution="close",
    ),
    "6": StrategyDefinition(
        name="MACD + RSI + MFI",
        description="Compra: MACD alcista + RSI<40 + MFI<40 | Venta: MACD bajista + RSI>60 + MFI>60",
        entry_fn=strategy_macd_rsi_mfi,
        exit_fn=strategy_macd_rsi_mfi_exit,
    ),
    "7": StrategyDefinition(
        name="Stoch RSI + MACD",
        description="Compra: StochRSI K bajo + MACD sobre signal | Venta: MACD bajo signal",
        entry_fn=strategy_stochrsi_macd,
        exit_fn=strategy_stochrsi_macd_exit,
    ),
    "8": StrategyDefinition(
        name="EMA50/200 + ADX + RSI",
        description="Golden/Death Cross con ADX>20 y RSI 30-70, SL/TP por ATR",
        entry_fn=strategy_ema50_200_adx_rsi,
        exit_fn=strategy_ema50_200_adx_rsi_exit,
        stop_atr=2.0,
        take_atr=4.0,
    ),
    "9": StrategyDefinition(
        name="ADX + EMA direccion",
        description="Compra: ADX minimo + Close>EMA14 | Venta: Close<EMA14",
        entry_fn=strategy_adx_ema_direction,
        exit_fn=strategy_adx_ema_direction_exit,
    ),
}
