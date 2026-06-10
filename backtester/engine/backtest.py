from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import borrow_costs, transaction_costs


@dataclass
class BacktestResult:
    returns_gross: pd.Series      # retorno diario bruto del portfolio
    returns_net: pd.Series        # retorno diario neto de costos
    equity_gross: pd.Series       # curva de equity bruta (empieza en 1.0)
    equity_net: pd.Series         # curva de equity neta
    weights: pd.DataFrame         # pesos efectivos (shifteados) por día
    turnover: pd.Series           # Σ|Δw| diario
    costs: pd.Series              # costo total diario (comisión + borrow)

    @property
    def annual_turnover(self) -> float:
        if self.turnover.empty:
            return 0.0
        years = max((self.turnover.index[-1] - self.turnover.index[0]).days / 365.25, 1e-9)
        return float(self.turnover.sum() / years)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    commission_bps: float = 10.0,
    short_borrow_bps_annual: float = 0.0,
    periods_per_year: float = 252.0,
) -> BacktestResult:
    """Motor vectorizado común a todas las estrategias.

    Contrato (plan §4):
        retorno_portfolio(t) = Σᵢ wᵢ(t-1) · rᵢ(t) - costos(Δw)

    - `weights`: pesos OBJETIVO calculados con datos hasta t (fechas × tickers).
      El motor los shiftea un día (regla anti-look-ahead, innegociable): el peso
      decidido con el close de t-1 se aplica al retorno de t.
    - `prices`: precios ajustados (fechas × tickers). Se reindexan a la grilla
      de `weights`.

    Las posiciones faltantes se tratan como 0 (sin exposición).
    """
    if weights.empty:
        raise ValueError("weights vacío: la estrategia no produjo señales.")

    # Retornos diarios alineados a las columnas de pesos.
    rets = prices.pct_change()

    # Alinear universo: pesos y retornos sobre las mismas columnas/fechas.
    cols = weights.columns
    rets = rets.reindex(index=weights.index, columns=cols)

    # Anti-look-ahead: el peso de t-1 gana el retorno de t.
    held = weights.shift(1).fillna(0.0)

    # Retorno bruto: si un retorno es NaN (activo aún sin historia) y su peso es 0,
    # no aporta. Forzamos contribución 0 donde el peso es 0 para evitar NaN*0.
    contrib = held * rets.fillna(0.0)
    returns_gross = contrib.sum(axis=1)

    # Costos sobre los pesos efectivamente sostenidos.
    tc = transaction_costs(held, commission_bps=commission_bps)
    bc = (
        borrow_costs(
            held,
            short_borrow_bps_annual=short_borrow_bps_annual,
            periods_per_year=periods_per_year,
        )
        if short_borrow_bps_annual > 0
        else pd.Series(0.0, index=held.index)
    )
    total_costs = tc.add(bc, fill_value=0.0)

    returns_net = returns_gross - total_costs

    equity_gross = (1.0 + returns_gross).cumprod()
    equity_net = (1.0 + returns_net).cumprod()

    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()

    return BacktestResult(
        returns_gross=returns_gross,
        returns_net=returns_net,
        equity_gross=equity_gross,
        equity_net=equity_net,
        weights=held,
        turnover=turnover,
        costs=total_costs,
    )
