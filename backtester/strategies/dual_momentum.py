from __future__ import annotations

import pandas as pd

from .base import Strategy, expand_to_daily, rebalance_dates


class DualMomentum(Strategy):
    """§4.1.2 — Rotación sectorial con doble momentum.

    Cada mes:
      1. Momentum relativo: rankear los 11 SPDR sectoriales por retorno
         acumulado de `lookback_months`.
      2. Momentum absoluto: comprar los top `top_n` sectores SOLO si
         SPY > MA(abs_ma_days). Si no, rotar al safe asset de mayor momentum.
    Long-only, equal-weight entre los seleccionados, Σw = 1.
    """

    name = "dual_momentum"
    has_short = False

    def __init__(
        self,
        sectors: list[str],
        benchmark: str,
        safe_assets: list[str],
        *,
        lookback_months: int = 6,
        top_n: int = 3,
        abs_ma_days: int = 200,
        rebalance: str = "M",
    ):
        self.sectors = sectors
        self.benchmark = benchmark
        self.safe_assets = safe_assets
        self.lookback_months = lookback_months
        self.top_n = top_n
        self.abs_ma_days = abs_ma_days
        self.rebalance = rebalance

    def required_tickers(self) -> list[str]:
        return list(dict.fromkeys([*self.sectors, self.benchmark, *self.safe_assets]))

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = self.required_tickers()
        prices = prices[[c for c in cols if c in prices.columns]]
        # Sectores/refugios efectivamente disponibles (algunos ETFs no existen
        # en períodos tempranos: XLRE desde 2015, XLC desde 2018).
        sectors = [s for s in self.sectors if s in prices.columns]
        safe_assets = [a for a in self.safe_assets if a in prices.columns]
        lookback_days = self.lookback_months * 21
        bench_ma = prices[self.benchmark].rolling(self.abs_ma_days).mean()

        reb = rebalance_dates(prices.index, self.rebalance)
        target = pd.DataFrame(0.0, index=reb, columns=prices.columns)

        for dt in reb:
            window = prices.loc[:dt]
            if len(window) < max(lookback_days, self.abs_ma_days) + 1:
                continue
            mom = window.iloc[-1] / window.iloc[-1 - lookback_days] - 1.0

            regime_on = window[self.benchmark].iloc[-1] > bench_ma.loc[dt]

            if regime_on:
                sector_mom = mom[sectors].dropna()
                if sector_mom.empty:
                    continue
                winners = sector_mom.sort_values(ascending=False).head(self.top_n).index
                if len(winners) == 0:
                    continue
                w = 1.0 / len(winners)
                for tk in winners:
                    target.loc[dt, tk] = w
            else:
                safe_mom = mom[safe_assets].dropna()
                if safe_mom.empty:
                    continue
                refuge = safe_mom.idxmax()
                target.loc[dt, refuge] = 1.0

        return expand_to_daily(target, prices.index)
