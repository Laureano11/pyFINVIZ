from __future__ import annotations

import copy
from itertools import product

import pandas as pd

from ..analysis.metrics import compute_metrics
from ..engine.backtest import run_backtest
from ..runner import build_strategy, load_configs
from ..data.loader import load_prices


def parameter_grid(
    strategy_name: str,
    grid: dict[str, list],
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Análisis de sensibilidad (plan §5.6): corre una estrategia sobre un grid
    de parámetros y devuelve una tabla con las métricas clave por combinación.

    `grid` mapea nombre-de-parámetro → lista de valores a probar. Solo se varían
    los parámetros presentes en `grid`; el resto sale de strategies.yaml.

    Si una estrategia solo funciona con UNA combinación, está overfitteada
    (plan §5.6) — esta tabla lo hace visible.
    """
    universe, params = load_configs()
    data_cfg = universe["data"]
    start = start or data_cfg["start"]
    end = end or data_cfg["end"]
    costs = params["costs"]

    keys = list(grid.keys())
    rows = []

    for combo in product(*(grid[k] for k in keys)):
        params_v = copy.deepcopy(params)
        for k, v in zip(keys, combo):
            params_v[strategy_name][k] = v

        strat = build_strategy(strategy_name, universe, params_v)
        tickers = list(dict.fromkeys([*strat.required_tickers(), *data_cfg["always"]]))
        prices = load_prices(tickers, start, end)
        if prices.empty:
            continue

        weights = strat.generate_weights(prices)
        borrow = costs["short_borrow_bps_annual"] if strat.has_short else 0.0
        res = run_backtest(
            weights,
            prices,
            commission_bps=costs["commission_bps"],
            short_borrow_bps_annual=borrow,
        )
        spy = prices["SPY"].pct_change() if "SPY" in prices.columns else None
        m = compute_metrics(res.returns_net, benchmark=spy, turnover_annual=res.annual_turnover)

        row = {k: v for k, v in zip(keys, combo)}
        row.update({
            "CAGR_%": round(m["CAGR"] * 100, 2),
            "Sharpe": round(m["Sharpe"], 3),
            "MaxDD_%": round(m["MaxDD"] * 100, 2),
            "Turnover": round(m.get("TurnoverAnnual", 0.0), 2),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def walk_forward(
    strategy_name: str,
    *,
    is_start: str = "2005-01-01",
    is_end: str = "2015-12-31",
    oos_start: str = "2016-01-01",
    oos_end: str = "2026-06-09",
) -> pd.DataFrame:
    """Walk-forward simple (plan §5.5): compara métricas in-sample vs out-of-sample
    con los parámetros del YAML. Nunca reportar solo in-sample.
    """
    universe, params = load_configs()
    costs = params["costs"]
    data_cfg = universe["data"]

    strat = build_strategy(strategy_name, universe, params)
    tickers = list(dict.fromkeys([*strat.required_tickers(), *data_cfg["always"]]))

    rows = []
    for label, s, e in [("in_sample", is_start, is_end), ("out_of_sample", oos_start, oos_end)]:
        prices = load_prices(tickers, s, e)
        if prices.empty:
            continue
        weights = strat.generate_weights(prices)
        borrow = costs["short_borrow_bps_annual"] if strat.has_short else 0.0
        res = run_backtest(
            weights, prices,
            commission_bps=costs["commission_bps"],
            short_borrow_bps_annual=borrow,
        )
        spy = prices["SPY"].pct_change() if "SPY" in prices.columns else None
        m = compute_metrics(res.returns_net, benchmark=spy, turnover_annual=res.annual_turnover)
        rows.append({
            "period": label, "from": s, "to": e,
            "CAGR_%": round(m["CAGR"] * 100, 2),
            "Sharpe": round(m["Sharpe"], 3),
            "MaxDD_%": round(m["MaxDD"] * 100, 2),
        })

    return pd.DataFrame(rows)
