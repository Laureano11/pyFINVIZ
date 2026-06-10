from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Últimos días de trading de cada período (M=mensual, W=semanal).

    Devuelve las fechas reales del índice donde se decide rebalanceo. La señal
    se calcula con datos hasta ese día; el motor la shiftea al día siguiente.
    """
    s = pd.Series(index, index=index)
    if freq == "M":
        grouped = s.groupby([index.year, index.month])
    elif freq == "W":
        iso = index.isocalendar()
        grouped = s.groupby([iso.year.values, iso.week.values])
    else:
        raise ValueError(f"freq no soportada: {freq}")
    return pd.DatetimeIndex(grouped.last().values).sort_values()


def expand_to_daily(
    target: pd.DataFrame, full_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Lleva pesos definidos solo en fechas de rebalanceo a la grilla diaria
    completa con forward-fill (se mantiene la posición entre rebalanceos)."""
    return target.reindex(full_index).ffill().fillna(0.0)


class Strategy(ABC):
    """Contrato central (plan §4): toda estrategia produce un DataFrame de pesos
    objetivo (fechas × tickers, sum|w| ≤ 1).

    Los pesos se calculan con datos hasta t; el motor los shiftea (no la estrategia).
    Cada estrategia declara los tickers que necesita en `required_tickers`.
    """

    name: str = "base"

    @abstractmethod
    def required_tickers(self) -> list[str]:
        """Tickers que la estrategia necesita descargar."""

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Devuelve pesos objetivo (fechas × tickers)."""

    # ¿La estrategia tiene lado corto? El motor cobra borrow solo si True.
    has_short: bool = False


class BuyAndHold(Strategy):
    """Estrategia de control: 100% en un solo ticker, sin rebalanceo.

    Sirve para el test del motor (plan Fase 0): buy & hold SPY debe replicar
    el retorno de SPY exacto.
    """

    name = "buy_and_hold"

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker

    def required_tickers(self) -> list[str]:
        return [self.ticker]

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=prices.index, columns=[self.ticker])
        w[self.ticker] = 1.0
        return w
