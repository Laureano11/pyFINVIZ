"""Tests del motor de backtest (plan Fase 0).

Test clave: buy & hold de un ticker debe replicar el retorno de ese ticker exacto
(salvo el desfase de 1 día por el shift anti-look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine.backtest import run_backtest
from backtester.strategies.base import BuyAndHold


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.RandomState(42)
    spy = 100 * (1 + pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)).cumprod()
    other = 50 * (1 + pd.Series(rng.normal(0.0002, 0.015, len(idx)), index=idx)).cumprod()
    return pd.DataFrame({"SPY": spy, "AAA": other})


def test_buy_and_hold_replicates_asset_exactly(synthetic_prices):
    """Buy&hold SPY sin costos = retornos de SPY (desde que el book está armado)."""
    strat = BuyAndHold("SPY")
    weights = strat.generate_weights(synthetic_prices[["SPY"]])
    res = run_backtest(weights, synthetic_prices, commission_bps=0.0)

    spy_ret = synthetic_prices["SPY"].pct_change()

    # El peso entra en t=0, se shiftea a t=1, gana el retorno desde t=1 en adelante.
    port = res.returns_gross.iloc[1:]
    expected = spy_ret.iloc[1:]

    np.testing.assert_allclose(port.values, expected.values, atol=1e-12)


def test_buy_and_hold_equity_matches_price_ratio(synthetic_prices):
    """La equity neta de buy&hold sin costos debe seguir el ratio de precios."""
    strat = BuyAndHold("SPY")
    weights = strat.generate_weights(synthetic_prices[["SPY"]])
    res = run_backtest(weights, synthetic_prices, commission_bps=0.0)

    spy = synthetic_prices["SPY"]
    # equity en t = precio(t)/precio(t0) donde t0 es el primer día con exposición (índice 1).
    price_ratio = spy / spy.iloc[1]
    equity = res.equity_gross / res.equity_gross.iloc[1]

    aligned = equity.iloc[1:]
    expected = price_ratio.iloc[1:]
    np.testing.assert_allclose(aligned.values, expected.values, atol=1e-9)


def test_costs_only_on_turnover(synthetic_prices):
    """Sin rebalanceo (buy&hold), solo se cobra costo el primer día (armado del book)."""
    strat = BuyAndHold("SPY")
    weights = strat.generate_weights(synthetic_prices[["SPY"]])
    res = run_backtest(weights, synthetic_prices, commission_bps=10.0)

    # El book se arma cuando held pasa de 0→1, o sea en el índice 1 (por el shift).
    # Ese único turnover cobra 10 bps sobre exposición 1.0 = 0.001. El resto ~0.
    assert res.costs.iloc[1] == pytest.approx(10 / 1e4, abs=1e-12)
    assert res.costs.drop(res.costs.index[1]).sum() == pytest.approx(0.0, abs=1e-15)


def test_dollar_neutral_has_zero_net_exposure():
    """Un book dólar-neutral (long A, short B) tiene exposición neta 0."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {"A": np.linspace(100, 110, 10), "B": np.linspace(100, 90, 10)}, index=idx
    )
    weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
    res = run_backtest(weights, prices, commission_bps=0.0)
    # Exposición neta de los pesos sostenidos = 0 en cada fila (tras el shift).
    net = res.weights.sum(axis=1)
    assert np.allclose(net.values, 0.0)


def test_shift_prevents_lookahead():
    """El retorno del día t usa el peso de t-1, no el de t."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [100, 100, 110, 110, 110]}, index=idx)
    # Peso = 1 solo el día 2 (índice 2), cuando A ya saltó. Sin shift capturaría
    # el salto; con shift no, porque el peso se aplica al retorno del día 3 (=0).
    weights = pd.DataFrame({"A": [0, 0, 1, 0, 0]}, index=idx, dtype=float)
    res = run_backtest(weights, prices, commission_bps=0.0)
    # El salto de precio (día 1→2: +10%) NO debe ser capturado.
    assert res.returns_gross.iloc[2] == pytest.approx(0.0)
    # El peso de t=2 gana el retorno de t=3 (que es 0 acá).
    assert res.returns_gross.iloc[3] == pytest.approx(0.0)
