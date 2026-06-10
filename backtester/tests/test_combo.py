"""Tests del combo de estrategias (§3.20/§6).

El combo se construye sobre retornos diarios; probamos las funciones puras de
combinación con series sintéticas para no depender de la red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.analysis.combo import (
    _equal_weight,
    _risk_parity,
    mean_pairwise_correlation,
)
from backtester.analysis.metrics import compute_metrics


def _two_uncorrelated_strategies(days=1000, seed=0):
    idx = pd.date_range("2018-01-01", periods=days, freq="B")
    rng = np.random.RandomState(seed)
    a = pd.Series(rng.normal(0.0004, 0.01, days), index=idx)
    b = pd.Series(rng.normal(0.0004, 0.01, days), index=idx)  # independiente
    return pd.DataFrame({"A": a, "B": b})


def test_equal_weight_is_simple_average():
    rdf = _two_uncorrelated_strategies()
    combo = _equal_weight(rdf)
    expected = (rdf["A"] + rdf["B"]) / 2
    np.testing.assert_allclose(combo.values, expected.values, atol=1e-12)


def test_diversification_improves_sharpe():
    """Dos estrategias independientes con Sharpe positivo similar: el combo
    equal-weight debe mejorar el Sharpe ~sqrt(2) sobre cada parte (diversificación).

    Construimos drift fuerte y baja varianza para que ambas tengan Sharpe
    claramente positivo (evita el azar de que una serie salga negativa)."""
    idx = pd.date_range("2018-01-01", periods=2000, freq="B")
    rng = np.random.RandomState(7)
    # drift/vol idénticos → mismo Sharpe esperado, ruido independiente.
    a = pd.Series(rng.normal(0.0006, 0.006, 2000), index=idx)
    b = pd.Series(rng.normal(0.0006, 0.006, 2000), index=idx)
    rdf = pd.DataFrame({"A": a, "B": b})
    sharpe_a = compute_metrics(rdf["A"])["Sharpe"]
    sharpe_b = compute_metrics(rdf["B"])["Sharpe"]
    sharpe_combo = compute_metrics(_equal_weight(rdf))["Sharpe"]
    assert sharpe_a > 0 and sharpe_b > 0          # premisa: ambas positivas
    assert sharpe_combo > max(sharpe_a, sharpe_b)  # diversificación mejora


def test_risk_parity_no_lookahead():
    """Los pesos del risk-parity se shiftean: un shock futuro no cambia el
    retorno del combo en fechas previas."""
    rdf = _two_uncorrelated_strategies(seed=5)
    combo1 = _risk_parity(rdf, vol_lookback=63)
    cutoff = int(len(rdf) * 0.95)
    rdf2 = rdf.copy()
    rdf2.iloc[cutoff:] *= 3.0
    combo2 = _risk_parity(rdf2, vol_lookback=63)
    pd.testing.assert_series_equal(
        combo1.iloc[:cutoff], combo2.iloc[:cutoff], check_exact=False, atol=1e-12
    )


def test_risk_parity_weights_inverse_to_vol():
    """La pata más volátil debe recibir menos peso en risk-parity."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    rng = np.random.RandomState(1)
    calm = pd.Series(rng.normal(0.0003, 0.005, 400), index=idx)   # baja vol
    wild = pd.Series(rng.normal(0.0003, 0.02, 400), index=idx)    # alta vol
    rdf = pd.DataFrame({"calm": calm, "wild": wild})
    combo = _risk_parity(rdf, vol_lookback=63)
    # El combo debe parecerse más a la pata calma (que recibe más peso).
    corr_calm = combo.iloc[100:].corr(calm.iloc[100:])
    corr_wild = combo.iloc[100:].corr(wild.iloc[100:])
    assert corr_calm > corr_wild


def test_mean_pairwise_correlation():
    corr = pd.DataFrame(
        [[1.0, 0.2, 0.0], [0.2, 1.0, 0.4], [0.0, 0.4, 1.0]],
        index=["a", "b", "c"], columns=["a", "b", "c"],
    )
    # promedio de off-diagonal: (0.2+0.0+0.2+0.4+0.0+0.4)/6 = 0.2
    assert mean_pairwise_correlation(corr) == pytest.approx(0.2)
