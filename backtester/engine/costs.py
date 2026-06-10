from __future__ import annotations

import pandas as pd


def transaction_costs(
    weights: pd.DataFrame,
    *,
    commission_bps: float,
) -> pd.Series:
    """Costo de comisión por período = bps * turnover, donde turnover = Σ|Δw|.

    weights: pesos ya shifteados (los efectivamente sostenidos en cada t).
    Devuelve una Serie de costos en fracción del capital, alineada al índice.
    """
    turnover = weights.diff().abs().sum(axis=1)
    # El primer período arma todo el book desde cash: turnover = Σ|w| inicial.
    first = weights.iloc[0].abs().sum()
    turnover.iloc[0] = first
    return turnover * (commission_bps / 1e4)


def borrow_costs(
    weights: pd.DataFrame,
    *,
    short_borrow_bps_annual: float,
    periods_per_year: float,
) -> pd.Series:
    """Costo de shorting = tasa anual prorrateada * exposición corta del período."""
    short_exposure = weights.clip(upper=0.0).abs().sum(axis=1)
    per_period_rate = (short_borrow_bps_annual / 1e4) / periods_per_year
    return short_exposure * per_period_rate
