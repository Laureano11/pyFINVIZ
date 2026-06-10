"""Tests de las estrategias cross-sectional (§3.1 momentum, §3.9 mean reversion)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine.backtest import run_backtest
from backtester.strategies.mean_reversion import MeanReversion
from backtester.strategies.momentum_xs import MomentumXS


def _universe(tickers, days=500, seed=1):
    idx = pd.date_range("2018-01-01", periods=days, freq="B")
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {t: 100 * (1 + pd.Series(rng.normal(0.0003, 0.015, days), index=idx)).cumprod()
         for t in tickers}
    )


def test_momentum_xs_dollar_neutral():
    tickers = [f"S{i}" for i in range(40)]
    px = _universe(tickers, days=400)
    strat = MomentumXS(tickers, lookback_months=12, skip_months=1, long_short=True, min_names=10)
    w = strat.generate_weights(px)
    net = w.sum(axis=1)
    active = net[w.abs().sum(axis=1) > 0.01]
    # Dólar-neutral: exposición neta ≈ 0 en filas activas.
    assert np.allclose(active.values, 0.0, atol=1e-9)
    # Exposición bruta ≈ 1 (Σ|w| = 1).
    gross = w.abs().sum(axis=1)
    active_gross = gross[gross > 0.01]
    assert np.allclose(active_gross.values, 1.0, atol=1e-9)


def test_momentum_xs_long_only_positive():
    tickers = [f"S{i}" for i in range(40)]
    px = _universe(tickers, days=400)
    strat = MomentumXS(tickers, lookback_months=12, long_short=False, min_names=10)
    w = strat.generate_weights(px)
    assert (w >= 0).all().all()
    sums = w.sum(axis=1)
    active = sums[sums > 0.01]
    assert np.allclose(active.values, 1.0, atol=1e-9)


def test_momentum_xs_no_lookahead():
    tickers = [f"S{i}" for i in range(30)]
    px = _universe(tickers, days=400)
    strat = MomentumXS(tickers, lookback_months=12, long_short=True, min_names=10)
    w1 = strat.generate_weights(px)
    cutoff = int(len(px) * 0.98)
    px2 = px.copy()
    px2.iloc[cutoff:] *= 1.5
    w2 = strat.generate_weights(px2)
    pd.testing.assert_frame_equal(w1.iloc[:cutoff], w2.iloc[:cutoff])


def test_mean_reversion_dollar_neutral_per_group():
    groups = {"a": ["A1", "A2", "A3", "A4"], "b": ["B1", "B2", "B3", "B4"]}
    tickers = [t for g in groups.values() for t in g]
    px = _universe(tickers, days=200)
    strat = MeanReversion(groups, lookback_days=5, rebalance="W")
    w = strat.generate_weights(px)
    net = w.sum(axis=1)
    active = net[w.abs().sum(axis=1) > 0.01]
    assert np.allclose(active.values, 0.0, atol=1e-9)


def test_mean_reversion_shorts_winners_longs_losers():
    """El activo que más subió debe quedar short; el que más cayó, long."""
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    # A sube fuerte, B cae fuerte, C y D planos.
    px = pd.DataFrame({
        "A": np.linspace(100, 130, 20),
        "B": np.linspace(100, 70, 20),
        "C": np.full(20, 100.0),
        "D": np.full(20, 100.0),
    }, index=idx)
    strat = MeanReversion({"g": ["A", "B", "C", "D"]}, lookback_days=5, rebalance="W")
    w = strat.generate_weights(px)
    last = w.iloc[-1]
    assert last["A"] < 0  # subió → short
    assert last["B"] > 0  # cayó → long


def test_mean_reversion_runs_through_engine():
    groups = {"a": ["A1", "A2", "A3"], "b": ["B1", "B2", "B3"]}
    tickers = [t for g in groups.values() for t in g]
    px = _universe(tickers, days=200)
    strat = MeanReversion(groups, lookback_days=5, rebalance="W")
    w = strat.generate_weights(px)
    res = run_backtest(w, px, commission_bps=10.0, short_borrow_bps_annual=35.0)
    assert res.equity_net.iloc[-1] > 0
