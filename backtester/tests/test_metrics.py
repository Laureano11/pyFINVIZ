"""Tests de métricas y del flujo de comparación."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.analysis.metrics import (
    beta_vs_benchmark,
    compute_metrics,
    metrics_table,
)


@pytest.fixture
def returns():
    idx = pd.date_range("2015-01-01", periods=1000, freq="B")
    rng = np.random.RandomState(7)
    return pd.Series(rng.normal(0.0005, 0.01, 1000), index=idx)


def test_metrics_keys(returns):
    m = compute_metrics(returns, benchmark=returns, turnover_annual=3.0)
    for k in ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "PctPosMonths"]:
        assert k in m
    assert m["TurnoverAnnual"] == 3.0


def test_beta_self_is_one(returns):
    assert beta_vs_benchmark(returns, returns) == pytest.approx(1.0, abs=1e-9)


def test_maxdd_is_negative_or_zero(returns):
    m = compute_metrics(returns)
    assert m["MaxDD"] <= 0


def test_pct_positive_months_in_range(returns):
    m = compute_metrics(returns)
    assert 0.0 <= m["PctPosMonths"] <= 100.0


def test_known_sharpe():
    """Retorno constante diario → Sharpe muy alto, vol casi 0."""
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    r = pd.Series(0.001, index=idx)
    m = compute_metrics(r)
    assert m["Vol"] == pytest.approx(0.0, abs=1e-9)
    # CAGR ≈ (1.001)^252 - 1
    assert m["CAGR"] == pytest.approx((1.001 ** 252) - 1, rel=1e-6)


def test_metrics_table_formats_percentages():
    m = {"S": {"CAGR": 0.1, "Vol": 0.2, "Sharpe": 1.5, "MaxDD": -0.3,
              "Sortino": 2.0, "Calmar": 0.33, "PctPosMonths": 60.0}}
    table = metrics_table(m)
    assert table.loc["S", "CAGR"] == 10.0   # convertido a %
    assert table.loc["S", "Sharpe"] == 1.5  # ratio sin escalar
    assert table.loc["S", "PctPosMonths"] == 60.0  # ya en %, sin doble escala
