from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd

from pyfinviz.data_sources import get_market_history
from pyfinviz.indicators import compute_indicators
from pyfinviz.strategies import STRATEGIES


def _load_history(
    symbol: str,
    period: str,
    interval: str,
    market_data_source: str,
    market_data_fallback_to_yfinance: bool,
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_use_rth: bool,
    ibkr_what_to_show: str,
) -> pd.DataFrame:
    return get_market_history(
        symbol=symbol,
        period=period,
        interval=interval,
        source=market_data_source,
        fallback_to_yfinance=market_data_fallback_to_yfinance,
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_use_rth=ibkr_use_rth,
        ibkr_what_to_show=ibkr_what_to_show,
    )


def analyze_strategy_after_n_days(
    symbol: str,
    period: str,
    interval: str,
    market_data_source: str,
    market_data_fallback_to_yfinance: bool,
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_use_rth: bool,
    ibkr_what_to_show: str,
    horizon_days: int,
    ma200_filter_mode: str,
    trade_capital_usd: float,
    strategy_key: str,
    strategy_params: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = _load_history(
        symbol=symbol,
        period=period,
        interval=interval,
        market_data_source=market_data_source,
        market_data_fallback_to_yfinance=market_data_fallback_to_yfinance,
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_use_rth=ibkr_use_rth,
        ibkr_what_to_show=ibkr_what_to_show,
    )
    if df.empty:
        return pd.DataFrame(), {}

    df = compute_indicators(df)
    strategy = STRATEGIES.get(strategy_key, STRATEGIES["1"])

    entry_signal = strategy.entry_fn(df, ma200_filter_mode, strategy_params)
    if strategy.exit_fn is not None:
        exit_signal = strategy.exit_fn(df, strategy_params)
    else:
        exit_signal = pd.Series(False, index=df.index)

    df["entry_signal"] = entry_signal.fillna(False)
    df["exit_signal"] = exit_signal.fillna(False)

    first_date = df.index.min()
    last_date = df.index.max()
    span_years = max((last_date - first_date).days / 365.25, 1 / 365.25)
    total_signals = int(df["entry_signal"].sum())
    avg_signals_per_year = float(total_signals / span_years)

    trades: list[dict[str, Any]] = []
    next_entry_allowed_idx = 0

    for i in range(len(df)):
        if i < next_entry_allowed_idx:
            continue

        if not bool(df.iloc[i]["entry_signal"]):
            continue

        # Global rule: all strategies enter at the signal day's close.
        entry_idx = i
        entry_price = float(df.iloc[entry_idx]["Close"])

        horizon_exit_idx = entry_idx + horizon_days
        if horizon_exit_idx >= len(df):
            continue

        exit_idx = horizon_exit_idx
        exit_price = float(df.iloc[exit_idx]["Close"])
        exit_reason = "horizon"
        signal_exit_idx: Optional[int] = None

        stop_atr = strategy.stop_atr
        take_atr = strategy.take_atr
        if stop_atr > 0 and take_atr > 0 and pd.notna(df.iloc[entry_idx]["ATR14"]):
            entry_atr = float(df.iloc[entry_idx]["ATR14"])
            stop_price = entry_price - stop_atr * entry_atr
            take_price = entry_price + take_atr * entry_atr
        else:
            stop_price = None
            take_price = None

        exit_search_start = entry_idx + 1
        max_signal_idx = min(horizon_exit_idx - 1, len(df) - 2)
        if max_signal_idx >= exit_search_start:
            exit_candidates = df.iloc[exit_search_start : max_signal_idx + 1]
            hit = exit_candidates[exit_candidates["exit_signal"]]
            if not hit.empty:
                first_hit_idx = df.index.get_loc(hit.index[0])
                if strategy.execution == "close":
                    signal_exit_idx = first_hit_idx
                    signal_exit_price = float(df.iloc[signal_exit_idx]["Close"])
                else:
                    signal_exit_idx = first_hit_idx + 1
                    if signal_exit_idx <= horizon_exit_idx:
                        signal_exit_price = float(df.iloc[signal_exit_idx]["Open"])
                    else:
                        signal_exit_idx = None

                if signal_exit_idx is not None:
                    exit_idx = signal_exit_idx
                    exit_price = signal_exit_price
                    exit_reason = "signal"

        if stop_price is not None and take_price is not None:
            # Stop/take can only trigger from the next bar after a close entry.
            risk_start_idx = entry_idx + 1
            for j in range(risk_start_idx, horizon_exit_idx + 1):
                bar_low = float(df.iloc[j]["Low"])
                bar_high = float(df.iloc[j]["High"])

                if bar_low <= stop_price:
                    risk_exit_idx = j
                    risk_exit_price = stop_price
                    risk_exit_reason = "stop_loss"
                elif bar_high >= take_price:
                    risk_exit_idx = j
                    risk_exit_price = take_price
                    risk_exit_reason = "take_profit"
                else:
                    continue

                if signal_exit_idx is not None and signal_exit_idx <= risk_exit_idx:
                    break

                exit_idx = risk_exit_idx
                exit_price = risk_exit_price
                exit_reason = risk_exit_reason
                break

        ret = (exit_price / entry_price) - 1.0
        shares = trade_capital_usd / entry_price
        final_usd = shares * exit_price
        pnl_usd = final_usd - trade_capital_usd

        trades.append(
            {
                "signal_date": df.index[i],
                "entry_date": df.index[entry_idx],
                "exit_date": df.index[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "holding_days": horizon_days,
                "roi_pct": ret * 100,
                "trade_capital_usd": trade_capital_usd,
                "final_usd": final_usd,
                "pnl_usd": pnl_usd,
            }
        )

        # Prevent duplicate overlapping trades from consecutive entry signals.
        next_entry_allowed_idx = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {
            "total_trades": 0,
            "total_signals": float(total_signals),
            "avg_signals_per_year": avg_signals_per_year,
            "avg_pnl_per_year_usd": 0.0,
            "positive_trades": 0,
            "negative_trades": 0,
            "neutral_trades": 0,
            "avg_roi_pct": 0.0,
            "total_invested_usd": 0.0,
            "total_final_usd": 0.0,
            "total_pnl_usd": 0.0,
            "avg_pnl_usd": 0.0,
        }

    stats = {
        "total_trades": float(len(trades_df)),
        "total_signals": float(total_signals),
        "avg_signals_per_year": avg_signals_per_year,
        "positive_trades": float((trades_df["roi_pct"] > 0).sum()),
        "negative_trades": float((trades_df["roi_pct"] < 0).sum()),
        "neutral_trades": float((trades_df["roi_pct"] == 0).sum()),
        "avg_roi_pct": float(trades_df["roi_pct"].mean()),
        "total_invested_usd": float(trades_df["trade_capital_usd"].sum()),
        "total_final_usd": float(trades_df["final_usd"].sum()),
        "total_pnl_usd": float(trades_df["pnl_usd"].sum()),
        "avg_pnl_usd": float(trades_df["pnl_usd"].mean()),
        "avg_pnl_per_year_usd": float(trades_df["pnl_usd"].sum() / span_years),
    }
    return trades_df, stats


def _batch_row(
    symbol: str,
    period: str,
    interval: str,
    market_data_source: str,
    market_data_fallback_to_yfinance: bool,
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_use_rth: bool,
    ibkr_what_to_show: str,
    horizon_days: int,
    ma200_filter_mode: str,
    trade_capital_usd: float,
    strategy_key: str,
    strategy_params: dict[str, float],
) -> dict[str, Any]:
    _, stats = analyze_strategy_after_n_days(
        symbol=symbol,
        period=period,
        interval=interval,
        market_data_source=market_data_source,
        market_data_fallback_to_yfinance=market_data_fallback_to_yfinance,
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_use_rth=ibkr_use_rth,
        ibkr_what_to_show=ibkr_what_to_show,
        horizon_days=horizon_days,
        ma200_filter_mode=ma200_filter_mode,
        trade_capital_usd=trade_capital_usd,
        strategy_key=strategy_key,
        strategy_params=strategy_params,
    )

    if not stats:
        return {
            "symbol": symbol,
            "total_signals": 0,
            "avg_signals_per_year": 0.0,
            "avg_pnl_per_year_usd": 0.0,
            "total_trades": 0,
            "positive_trades": 0,
            "negative_trades": 0,
            "avg_roi_pct": 0.0,
            "total_pnl_usd": 0.0,
        }

    return {
        "symbol": symbol,
        "total_signals": int(stats["total_signals"]),
        "avg_signals_per_year": round(float(stats["avg_signals_per_year"]), 2),
        "avg_pnl_per_year_usd": round(float(stats["avg_pnl_per_year_usd"]), 2),
        "total_trades": int(stats["total_trades"]),
        "positive_trades": int(stats["positive_trades"]),
        "negative_trades": int(stats["negative_trades"]),
        "avg_roi_pct": round(float(stats["avg_roi_pct"]), 2),
        "total_pnl_usd": round(float(stats["total_pnl_usd"]), 2),
    }


def analyze_symbols_batch(
    symbols: list[str],
    period: str,
    interval: str,
    market_data_source: str,
    market_data_fallback_to_yfinance: bool,
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_use_rth: bool,
    ibkr_what_to_show: str,
    horizon_days: int,
    ma200_filter_mode: str,
    trade_capital_usd: float,
    strategy_key: str,
    strategy_params: dict[str, float],
    batch_workers: int,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    workers = max(1, min(batch_workers, len(symbols)))
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _batch_row,
                symbol,
                period,
                interval,
                market_data_source,
                market_data_fallback_to_yfinance,
                ibkr_host,
                ibkr_port,
                ibkr_client_id,
                ibkr_use_rth,
                ibkr_what_to_show,
                horizon_days,
                ma200_filter_mode,
                trade_capital_usd,
                strategy_key,
                strategy_params,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                rows.append(
                    {
                        "symbol": symbol,
                        "total_signals": 0,
                        "avg_signals_per_year": 0.0,
                        "avg_pnl_per_year_usd": 0.0,
                        "total_trades": 0,
                        "positive_trades": 0,
                        "negative_trades": 0,
                        "avg_roi_pct": 0.0,
                        "total_pnl_usd": 0.0,
                    }
                )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(by="total_pnl_usd", ascending=False).reset_index(drop=True)


def _scan_symbol_entries(
    symbol: str,
    period: str,
    interval: str,
    market_data_source: str,
    market_data_fallback_to_yfinance: bool,
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_use_rth: bool,
    ibkr_what_to_show: str,
    ma200_filter_mode: str,
    strategy_params: dict[str, float],
) -> list[dict[str, Any]]:
    df = _load_history(
        symbol=symbol,
        period=period,
        interval=interval,
        market_data_source=market_data_source,
        market_data_fallback_to_yfinance=market_data_fallback_to_yfinance,
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_use_rth=ibkr_use_rth,
        ibkr_what_to_show=ibkr_what_to_show,
    )
    if df.empty:
        return []

    df = compute_indicators(df)
    signal_date = df.index[-1]
    close_price = float(df.iloc[-1]["Close"])
    atr14_last = df.iloc[-1]["ATR14"]
    rows: list[dict[str, Any]] = []

    for key, strategy in STRATEGIES.items():
        signal = strategy.entry_fn(df, ma200_filter_mode, strategy_params).fillna(False)
        if bool(signal.iloc[-1]):
            tp_price: Optional[float] = None
            if strategy.take_atr > 0 and pd.notna(atr14_last):
                tp_price = close_price + (strategy.take_atr * float(atr14_last))

            rows.append(
                {
                    "symbol": symbol,
                    "strategy_key": key,
                    "strategy_name": strategy.name,
                    "signal_date": signal_date,
                    "entry_price": round(close_price, 4),
                    "tp_price": round(tp_price, 4) if tp_price is not None else None,
                }
            )

    return rows


def scan_entry_opportunities(
    symbols: list[str],
    period: str,
    interval: str,
    market_data_source: str,
    market_data_fallback_to_yfinance: bool,
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_use_rth: bool,
    ibkr_what_to_show: str,
    ma200_filter_mode: str,
    strategy_params: dict[str, float],
    batch_workers: int,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    workers = max(1, min(batch_workers, len(symbols)))
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _scan_symbol_entries,
                symbol,
                period,
                interval,
                market_data_source,
                market_data_fallback_to_yfinance,
                ibkr_host,
                ibkr_port,
                ibkr_client_id,
                ibkr_use_rth,
                ibkr_what_to_show,
                ma200_filter_mode,
                strategy_params,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                continue

    opportunities_df = pd.DataFrame(rows)
    if opportunities_df.empty:
        return opportunities_df

    return opportunities_df.sort_values(
        by=["signal_date", "symbol", "strategy_key"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
