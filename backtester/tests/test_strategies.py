"""Tests de las estrategias: contrato de pesos y ausencia de look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine.backtest import run_backtest
from backtester.strategies.base import rebalance_dates
from backtester.strategies.dual_momentum import DualMomentum
from backtester.strategies.trend_following import TrendFollowing


def _synthetic_universe(tickers, days=900, seed=0):
    idx = pd.date_range("2018-01-01", periods=days, freq="B")
    rng = np.random.RandomState(seed)
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 + 0.0001 * i
        data[t] = 100 * (1 + pd.Series(rng.normal(drift, 0.012, days), index=idx)).cumprod()
    return pd.DataFrame(data)


def test_rebalance_dates_monthly_count():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    reb = rebalance_dates(idx, "M")
    assert len(reb) == 12  # 12 meses


def test_dual_momentum_weights_sum_to_one_or_zero():
    sectors = ["XLK", "XLF", "XLV", "XLE", "XLY"]
    px = _synthetic_universe([*sectors, "SPY", "TLT", "GLD"])
    strat = DualMomentum(sectors, "SPY", ["TLT", "GLD"], lookback_months=6, top_n=3)
    w = strat.generate_weights(px)
    sums = w.sum(axis=1)
    # En cada fila Σw es 0 (sin señal aún) o ≈1 (long-only, fully invested).
    valid = sums[(sums > 0.01)]
    assert np.allclose(valid.values, 1.0, atol=1e-9)
    assert (w >= 0).all().all()  # long-only


def test_dual_momentum_no_lookahead():
    """Inyectar un salto futuro no debe cambiar el peso decidido antes del salto."""
    sectors = ["XLK", "XLF", "XLV", "XLE", "XLY"]
    px = _synthetic_universe([*sectors, "SPY", "TLT", "GLD"])
    strat = DualMomentum(sectors, "SPY", ["TLT", "GLD"], lookback_months=6, top_n=3)

    w1 = strat.generate_weights(px)
    # Modificar solo el último 1% de las fechas (futuro).
    cutoff = int(len(px) * 0.99)
    px2 = px.copy()
    px2.iloc[cutoff:] *= 2.0
    w2 = strat.generate_weights(px2)
    # Los pesos antes del cutoff deben ser idénticos (no miran el futuro).
    pd.testing.assert_frame_equal(w1.iloc[:cutoff], w2.iloc[:cutoff])


def test_trend_following_long_only_and_normalized():
    assets = ["SPY", "EFA", "EEM", "TLT", "GLD", "DBC"]
    px = _synthetic_universe(assets)
    strat = TrendFollowing(assets, cash_proxy="SHY", lookback_months=6)
    w = strat.generate_weights(px)
    assert (w >= 0).all().all()
    sums = w.sum(axis=1)
    valid = sums[sums > 0.01]
    assert np.allclose(valid.values, 1.0, atol=1e-9)


def test_dual_momentum_handles_missing_sector_columns():
    """Si un ETF sectorial no existe en el período (p.ej. XLC antes de 2018),
    la estrategia no debe romper: opera con los sectores disponibles."""
    sectors = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLC"]  # XLC ausente abajo
    present = ["XLK", "XLF", "XLV", "XLE", "XLY", "SPY", "TLT", "GLD"]
    px = _synthetic_universe(present)  # sin XLC
    strat = DualMomentum(sectors, "SPY", ["TLT", "GLD"], lookback_months=6, top_n=3)
    w = strat.generate_weights(px)  # no debe lanzar KeyError
    assert "XLC" not in w.columns
    sums = w.sum(axis=1)
    valid = sums[sums > 0.01]
    assert np.allclose(valid.values, 1.0, atol=1e-9)


def test_strategies_runnable_through_engine():
    sectors = ["XLK", "XLF", "XLV", "XLE", "XLY"]
    px = _synthetic_universe([*sectors, "SPY", "TLT", "GLD"])
    strat = DualMomentum(sectors, "SPY", ["TLT", "GLD"], lookback_months=6, top_n=3)
    w = strat.generate_weights(px)
    res = run_backtest(w, px, commission_bps=10.0)
    assert not res.returns_net.isna().all()
    assert res.equity_net.iloc[-1] > 0
