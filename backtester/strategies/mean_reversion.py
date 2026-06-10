from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, expand_to_daily, rebalance_dates


class MeanReversion(Strategy):
    """§3.9 — Reversión a la media, grupo único (por sector).

    Dentro de cada grupo de acciones correlacionadas (un sector), calcular el
    retorno neto de la media R̃ᵢ = Rᵢ - R̄ sobre los últimos `lookback_days`.
    Posiciones en dólares Dᵢ ∝ -R̃ᵢ (vender lo que subió de más, comprar lo que
    cayó de más), normalizando con γ para que Σ|Dᵢ| = gross_exposure
    (Ec. 295-298). Dólar-neutral por construcción dentro de cada grupo.
    """

    name = "mean_reversion"
    has_short = True

    def __init__(
        self,
        sector_groups: dict[str, list[str]],
        *,
        lookback_days: int = 5,
        rebalance: str = "W",
        gross_exposure: float = 1.0,
    ):
        self.sector_groups = sector_groups
        self.lookback_days = lookback_days
        self.rebalance = rebalance
        self.gross_exposure = gross_exposure

    def required_tickers(self) -> list[str]:
        out: list[str] = []
        for names in self.sector_groups.values():
            out.extend(names)
        return list(dict.fromkeys(out))

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        all_names = [c for c in self.required_tickers() if c in prices.columns]
        prices = prices[all_names]

        reb = rebalance_dates(prices.index, self.rebalance)
        target = pd.DataFrame(0.0, index=reb, columns=prices.columns)

        n_groups = len(self.sector_groups)
        # Repartir la exposición bruta entre grupos por igual.
        per_group_gross = self.gross_exposure / max(n_groups, 1)

        for dt in reb:
            window = prices.loc[:dt]
            if len(window) < self.lookback_days + 1:
                continue

            for names in self.sector_groups.values():
                valid = [n for n in names if n in window.columns]
                if len(valid) < 2:
                    continue
                sub = window[valid]
                # Retorno acumulado de los últimos d días por activo.
                r = sub.iloc[-1] / sub.iloc[-1 - self.lookback_days] - 1.0
                r = r.dropna()
                if len(r) < 2:
                    continue
                # R̃ᵢ = Rᵢ - R̄ (demean dentro del grupo).
                r_tilde = r - r.mean()
                d = -r_tilde  # Dᵢ ∝ -R̃ᵢ
                denom = d.abs().sum()
                if denom == 0:
                    continue
                gamma = per_group_gross / denom
                for n in d.index:
                    target.loc[dt, n] += gamma * d[n]

        return expand_to_daily(target, prices.index)
