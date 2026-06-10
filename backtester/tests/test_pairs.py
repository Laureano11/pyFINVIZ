"""Tests de la estrategia de pares (§3.8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine.backtest import run_backtest
from backtester.strategies.pairs import PairsTrading


def _cointegrated_pair(days=600, seed=0):
    """Genera A y B cointegrados: B = random walk, A = B + ruido estacionario."""
    idx = pd.date_range("2018-01-01", periods=days, freq="B")
    rng = np.random.RandomState(seed)
    b = 100 + np.cumsum(rng.normal(0, 1, days))
    noise = rng.normal(0, 2, days)
    a = b + noise  # A y B comparten la tendencia → cointegrados
    # un tercer activo no relacionado
    c = 50 + np.cumsum(rng.normal(0, 1.5, days))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def test_pairs_selects_cointegrated_pair():
    px = _cointegrated_pair()
    strat = PairsTrading(
        {"g": ["A", "B", "C"]},
        formation_months=12,
        retest_months=3,
        pvalue_max=0.05,
        zscore_window=63,
        max_pairs=5,
    )
    window = px.iloc[-252:]
    pairs = strat._select_pairs(window)
    found = {(a, b) for a, b, _, _ in pairs}
    # A-B debe estar; C no cointegra con ninguno (es independiente).
    assert ("A", "B") in found


def test_pairs_dollar_neutral_when_active():
    px = _cointegrated_pair()
    strat = PairsTrading({"g": ["A", "B", "C"]}, zscore_window=40, entry_z=1.5, max_pairs=5)
    w = strat.generate_weights(px)
    net = w.sum(axis=1)
    active = net[w.abs().sum(axis=1) > 1e-9]
    # Cada posición es dólar-neutral (long una pata, short la otra en igual monto).
    assert np.allclose(active.values, 0.0, atol=1e-9)


def test_pairs_opens_positions():
    px = _cointegrated_pair()
    strat = PairsTrading({"g": ["A", "B", "C"]}, zscore_window=40, entry_z=1.5, max_pairs=5)
    w = strat.generate_weights(px)
    # Debe haber al menos algunos días con exposición (el spread oscila ±2σ).
    assert (w.abs().sum(axis=1) > 1e-9).sum() > 0


def test_pairs_no_lookahead():
    px = _cointegrated_pair()
    strat = PairsTrading({"g": ["A", "B", "C"]}, zscore_window=40, entry_z=1.5, max_pairs=5)
    w1 = strat.generate_weights(px)
    cutoff = int(len(px) * 0.95)
    px2 = px.copy()
    px2.iloc[cutoff:] += 50  # shock futuro
    w2 = strat.generate_weights(px2)
    # El estado de los pares re-forma en fechas fijas; antes del cutoff los pesos
    # no deben depender de datos posteriores al cutoff.
    pd.testing.assert_frame_equal(
        w1.iloc[:cutoff], w2.iloc[:cutoff], check_exact=False, atol=1e-9
    )


def test_pairs_runs_through_engine():
    px = _cointegrated_pair()
    strat = PairsTrading({"g": ["A", "B", "C"]}, zscore_window=40, entry_z=1.5, max_pairs=5)
    w = strat.generate_weights(px)
    res = run_backtest(w, px, commission_bps=10.0, short_borrow_bps_annual=35.0)
    assert res.equity_net.iloc[-1] > 0
