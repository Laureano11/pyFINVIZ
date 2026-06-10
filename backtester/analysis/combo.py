from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.metrics import compute_metrics, metrics_table
from .compare import ALL_STRATEGIES, run_all

TRADING_DAYS = 252.0


def _equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Combo equal-weight: promedio simple de los retornos netos diarios."""
    return returns.mean(axis=1)


def _risk_parity(returns: pd.DataFrame, vol_lookback: int = 63) -> pd.Series:
    """Combo risk-parity (inverse-vol): cada estrategia pesa ∝ 1/σ rolling, de
    modo que todas aporten un riesgo similar. Pesos decididos con datos hasta t-1
    (shift) y aplicados al retorno de t — misma regla anti-look-ahead del motor.
    """
    vol = returns.rolling(vol_lookback).std()
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.shift(1).fillna(0.0)
    return (weights * returns.fillna(0.0)).sum(axis=1)


def build_combo(
    strategies: list[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    method: str = "risk_parity",
    vol_lookback: int = 63,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construye el "combo alfa" (plan §3.20/§6): combina las estrategias en un
    solo portafolio. Como las correlaciones entre ellas son bajas, el combo suele
    tener mejor Sharpe que el promedio de sus partes.

    Devuelve (tabla de métricas [estrategias + combo + SPY], retorno del combo,
    matriz de correlaciones).
    """
    strategies = strategies or ALL_STRATEGIES
    _table, corr, results = run_all(strategies, start=start, end=end)

    net_returns = {name: res.returns_net for name, (res, _m, _s) in results.items()}
    rdf = pd.DataFrame(net_returns).dropna(how="all")

    spy = None
    for _name, (_res, _m, s) in results.items():
        if s is not None:
            spy = s
            break

    if method == "equal_weight":
        combo_ret = _equal_weight(rdf)
    elif method == "risk_parity":
        combo_ret = _risk_parity(rdf, vol_lookback=vol_lookback)
    else:
        raise ValueError(f"método de combo desconocido: {method}")

    # Métricas: cada estrategia + el combo + SPY benchmark.
    metrics_all: dict[str, dict] = {}
    for name, (res, _m, _s) in results.items():
        metrics_all[name] = compute_metrics(
            res.returns_net, benchmark=spy, turnover_annual=res.annual_turnover
        )
    metrics_all[f"COMBO_{method}"] = compute_metrics(combo_ret, benchmark=spy)
    if spy is not None:
        metrics_all["SPY_buyhold"] = compute_metrics(spy.dropna(), benchmark=spy)

    table = metrics_table(metrics_all)
    return table, combo_ret, corr


def mean_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Correlación promedio fuera de la diagonal — mide cuán diversificado es el set."""
    n = len(corr)
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(corr.values[mask].mean())
