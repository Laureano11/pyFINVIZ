from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, expand_to_daily, rebalance_dates


class TrendFollowing(Strategy):
    """§4.6 — Seguimiento de tendencia multi-activo.

    Cada mes, sobre ~10 ETFs de clases de activos distintas:
      1. Doble filtro: retorno acumulado `lookback_months` > 0 Y precio > MA(abs_ma_days).
      2. A los sobrevivientes, ponderación inverse-volatility (Ec. 372:
         wᵢ ∝ 1/σᵢ), normalizada a Σwᵢ = 1.
      3. Lo filtrado queda en cash (peso 0). Long-only.
    """

    name = "trend_following"
    has_short = False

    def __init__(
        self,
        assets: list[str],
        *,
        cash_proxy: str | None = None,
        lookback_months: int = 6,
        abs_ma_days: int = 200,
        vol_lookback: int = 63,
        rebalance: str = "M",
    ):
        self.assets = assets
        self.cash_proxy = cash_proxy
        self.lookback_months = lookback_months
        self.abs_ma_days = abs_ma_days
        self.vol_lookback = vol_lookback
        self.rebalance = rebalance

    def required_tickers(self) -> list[str]:
        out = list(self.assets)
        if self.cash_proxy:
            out.append(self.cash_proxy)
        return list(dict.fromkeys(out))

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        assets = [a for a in self.assets if a in prices.columns]
        prices = prices[assets]
        lookback_days = self.lookback_months * 21
        rets = prices.pct_change()
        ma = prices.rolling(self.abs_ma_days).mean()
        vol = rets.rolling(self.vol_lookback).std()

        reb = rebalance_dates(prices.index, self.rebalance)
        target = pd.DataFrame(0.0, index=reb, columns=prices.columns)

        for dt in reb:
            window = prices.loc[:dt]
            if len(window) < max(lookback_days, self.abs_ma_days) + 1:
                continue
            last = window.iloc[-1]
            mom = last / window.iloc[-1 - lookback_days] - 1.0

            above_ma = last > ma.loc[dt]
            positive_mom = mom > 0
            survivors = [
                a for a in assets
                if positive_mom.get(a, False) and above_ma.get(a, False)
                and not np.isnan(mom.get(a, np.nan))
            ]
            if not survivors:
                continue

            inv_vol = {}
            for a in survivors:
                v = vol.loc[dt, a]
                if v and not np.isnan(v) and v > 0:
                    inv_vol[a] = 1.0 / v
            if not inv_vol:
                # fallback equal-weight si no hay vol válida
                w = 1.0 / len(survivors)
                for a in survivors:
                    target.loc[dt, a] = w
                continue

            total = sum(inv_vol.values())
            for a, iv in inv_vol.items():
                target.loc[dt, a] = iv / total

        return expand_to_daily(target, prices.index)
