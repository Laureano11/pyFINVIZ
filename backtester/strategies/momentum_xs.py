from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, expand_to_daily, rebalance_dates


class MomentumXS(Strategy):
    """§3.1 — Precio-momentum cross-sectional.

    Cada mes, rankear las acciones por retorno acumulado de `lookback_months`
    salteando los últimos `skip_months` (12-1: evita la reversión de corto plazo).
    Long decil superior, short decil inferior (dólar-neutral), o long-only.

    Ponderación:
      - "uniform": ±1/(2·N) en cada lado (módulo-uniforme).
      - "inverse_vol": wᵢ ∝ 1/σᵢ dentro de cada lado.
    """

    name = "momentum_xs"

    def __init__(
        self,
        universe: list[str],
        *,
        lookback_months: int = 12,
        skip_months: int = 1,
        rebalance: str = "M",
        quantile: float = 0.1,
        weighting: str = "uniform",
        vol_lookback: int = 63,
        long_short: bool = True,
        min_names: int = 20,
    ):
        self.universe = universe
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.rebalance = rebalance
        self.quantile = quantile
        self.weighting = weighting
        self.vol_lookback = vol_lookback
        self.long_short = long_short
        self.min_names = min_names

    @property
    def has_short(self) -> bool:
        return self.long_short

    def required_tickers(self) -> list[str]:
        return list(self.universe)

    def _leg_weights(self, names: list[str], vol_row: pd.Series, sign: float) -> dict[str, float]:
        """Pesos de un lado (long o short) según el esquema de ponderación."""
        if not names:
            return {}
        if self.weighting == "inverse_vol":
            iv = {}
            for n in names:
                v = vol_row.get(n, np.nan)
                if v and not np.isnan(v) and v > 0:
                    iv[n] = 1.0 / v
            if iv:
                total = sum(iv.values())
                return {n: sign * 0.5 * (w / total) for n, w in iv.items()}
        # uniform (o fallback): ±1/(2N) por lado → Σ|w| = 1 con dos lados.
        w = sign * 0.5 / len(names)
        return {n: w for n in names}

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.universe if c in prices.columns]
        prices = prices[cols]
        lookback_days = self.lookback_months * 21
        skip_days = self.skip_months * 21
        rets = prices.pct_change()
        vol = rets.rolling(self.vol_lookback).std()

        reb = rebalance_dates(prices.index, self.rebalance)
        target = pd.DataFrame(0.0, index=reb, columns=prices.columns)

        # Posición entera de cada fecha de rebalanceo (evita copiar prices.loc[:dt]).
        positions = prices.index.get_indexer(reb)

        for dt, pos in zip(reb, positions):
            if pos < lookback_days:
                continue
            # Momentum 12-1: precio de hace skip_days / precio de hace lookback_days.
            p_recent = prices.iloc[pos - skip_days]
            p_old = prices.iloc[pos - lookback_days]
            mom = (p_recent / p_old - 1.0).dropna()
            if len(mom) < self.min_names:
                continue

            mom = mom.sort_values(ascending=False)
            n_leg = max(int(len(mom) * self.quantile), 1)
            longs = list(mom.head(n_leg).index)
            vol_row = vol.loc[dt]

            long_w = self._leg_weights(longs, vol_row, +1.0)
            for n, w in long_w.items():
                target.loc[dt, n] = w

            if self.long_short:
                shorts = list(mom.tail(n_leg).index)
                short_w = self._leg_weights(shorts, vol_row, -1.0)
                for n, w in short_w.items():
                    target.loc[dt, n] = w
            else:
                # long-only: el decil superior con Σw = 1.
                if longs:
                    if self.weighting == "inverse_vol":
                        iv = {n: 1.0 / vol_row[n] for n in longs
                              if vol_row.get(n, np.nan) and not np.isnan(vol_row.get(n, np.nan)) and vol_row[n] > 0}
                        if iv:
                            tot = sum(iv.values())
                            for n, v in iv.items():
                                target.loc[dt, n] = v / tot
                            continue
                    w = 1.0 / len(longs)
                    for n in longs:
                        target.loc[dt, n] = w

        return expand_to_daily(target, prices.index)
