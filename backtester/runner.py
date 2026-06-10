from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .analysis.metrics import compute_metrics
from .data.loader import load_prices
from .engine.backtest import BacktestResult, run_backtest
from .strategies.dual_momentum import DualMomentum
from .strategies.trend_following import TrendFollowing

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs() -> tuple[dict, dict]:
    return _load_yaml("universe.yaml"), _load_yaml("strategies.yaml")


def _read_universe_file(filename: str) -> list[str]:
    path = CONFIG_DIR / filename
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def build_strategy(name: str, universe: dict, params: dict):
    """Instancia una estrategia por nombre con sus parámetros."""
    if name == "dual_momentum":
        u = universe["dual_momentum"]
        p = params["dual_momentum"]
        return DualMomentum(
            sectors=u["sectors"],
            benchmark=u["benchmark"],
            safe_assets=u["safe_assets"],
            lookback_months=p["lookback_months"],
            top_n=p["top_n"],
            abs_ma_days=p["abs_ma_days"],
            rebalance=p["rebalance"],
        )
    if name == "trend_following":
        u = universe["trend_following"]
        p = params["trend_following"]
        return TrendFollowing(
            assets=u["assets"],
            cash_proxy=u.get("cash_proxy"),
            lookback_months=p["lookback_months"],
            abs_ma_days=p["abs_ma_days"],
            vol_lookback=p["vol_lookback"],
            rebalance=p["rebalance"],
        )
    if name == "momentum_xs":
        from .strategies.momentum_xs import MomentumXS

        p = params["momentum_xs"]
        tickers = _read_universe_file(universe["cross_sectional"]["universe_file"])
        return MomentumXS(
            universe=tickers,
            lookback_months=p["lookback_months"],
            skip_months=p["skip_months"],
            rebalance=p["rebalance"],
            quantile=p["quantile"],
            weighting=p["weighting"],
            vol_lookback=p["vol_lookback"],
            long_short=p["long_short"],
            min_names=p["min_names"],
        )
    if name == "mean_reversion":
        from .strategies.mean_reversion import MeanReversion

        p = params["mean_reversion"]
        sectors = universe["pairs"]["sectors"]
        return MeanReversion(
            sector_groups=sectors,
            lookback_days=p["lookback_days"],
            rebalance=p["rebalance"],
            gross_exposure=p["gross_exposure"],
        )
    if name == "pairs":
        from .strategies.pairs import PairsTrading

        p = params["pairs"]
        sectors = universe["pairs"]["sectors"]
        return PairsTrading(
            sector_groups=sectors,
            formation_months=p["formation_months"],
            retest_months=p["retest_months"],
            pvalue_max=p["pvalue_max"],
            zscore_window=p["zscore_window"],
            entry_z=p["entry_z"],
            exit_z=p["exit_z"],
            stop_z=p["stop_z"],
            time_stop_days=p["time_stop_days"],
            max_pairs=p["max_pairs"],
            gross_exposure=p["gross_exposure"],
        )
    raise ValueError(f"estrategia desconocida: {name}")


def run_strategy(
    name: str,
    *,
    universe: dict | None = None,
    params: dict | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> tuple[BacktestResult, dict, pd.Series]:
    """Corre una estrategia end-to-end. Devuelve (resultado, métricas, retorno SPY benchmark)."""
    if universe is None or params is None:
        universe, params = load_configs()

    data_cfg = universe["data"]
    start = start or data_cfg["start"]
    end = end or data_cfg["end"]

    strat = build_strategy(name, universe, params)
    tickers = list(dict.fromkeys([*strat.required_tickers(), *data_cfg["always"]]))

    prices = load_prices(tickers, start, end, refresh=refresh)
    if prices.empty:
        raise RuntimeError(f"sin datos para {name}")

    weights = strat.generate_weights(prices)

    costs = params["costs"]
    borrow = costs["short_borrow_bps_annual"] if strat.has_short else 0.0
    res = run_backtest(
        weights,
        prices,
        commission_bps=costs["commission_bps"],
        short_borrow_bps_annual=borrow,
    )

    spy_ret = prices["SPY"].pct_change() if "SPY" in prices.columns else None
    metrics = compute_metrics(
        res.returns_net,
        benchmark=spy_ret,
        turnover_annual=res.annual_turnover,
    )
    return res, metrics, spy_ret
