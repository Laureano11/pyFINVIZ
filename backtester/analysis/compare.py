from __future__ import annotations

import pandas as pd

from ..analysis.metrics import compute_metrics, metrics_table
from ..runner import load_configs, run_strategy

ALL_STRATEGIES = [
    "dual_momentum",
    "trend_following",
    "momentum_xs",
    "mean_reversion",
    "pairs",
]


def run_all(
    strategies: list[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Corre varias estrategias y devuelve (tabla de métricas, matriz de corr, results).

    `results[name]` = (BacktestResult, metrics, spy_ret).
    Incluye SPY buy&hold como benchmark en la tabla.
    """
    strategies = strategies or ALL_STRATEGIES
    universe, params = load_configs()

    results: dict = {}
    metrics_all: dict[str, dict] = {}
    net_returns: dict[str, pd.Series] = {}

    spy_ret_ref: pd.Series | None = None

    for name in strategies:
        try:
            res, m, spy = run_strategy(
                name, universe=universe, params=params, start=start, end=end, refresh=refresh
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[compare] {name} falló: {exc}")
            continue
        results[name] = (res, m, spy)
        metrics_all[name] = m
        net_returns[name] = res.returns_net
        if spy is not None and spy_ret_ref is None:
            spy_ret_ref = spy

    # Benchmark SPY buy & hold.
    if spy_ret_ref is not None:
        metrics_all["SPY_buyhold"] = compute_metrics(
            spy_ret_ref.dropna(), benchmark=spy_ret_ref
        )

    table = metrics_table(metrics_all)

    corr = pd.DataFrame()
    if net_returns:
        rdf = pd.DataFrame(net_returns).dropna(how="all")
        corr = rdf.corr()

    return table, corr, results
