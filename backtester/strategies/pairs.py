from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from .base import Strategy


class PairsTrading(Strategy):
    """§3.8 — Trading de pares con cointegración (Engle-Granger).

    Diseño event-driven sobre matriz de pesos:
      - Formación: cada `retest_months`, dentro de cada sector, testear coint.
        de todos los pares con ventana `formation_months`. Quedarse con
        p < `pvalue_max` (hasta `max_pairs`), ordenados por p-value.
      - Spread: residuo de OLS A ~ β·B + c. z-score rolling (`zscore_window`).
      - Estado por par:
          abrir cuando |z| ≥ entry_z (short la cara / long la barata),
          cerrar cuando z vuelve a la media (cruza exit_z), o |z| ≥ stop_z
          (stop por divergencia), o pasan `time_stop_days`.
      - Cada par operado usa exposición bruta `gross_exposure/max_pairs`,
        dólar-neutral (mitad long, mitad short).
    """

    name = "pairs"
    has_short = True

    def __init__(
        self,
        sector_groups: dict[str, list[str]],
        *,
        formation_months: int = 12,
        retest_months: int = 3,
        pvalue_max: float = 0.05,
        zscore_window: int = 63,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        stop_z: float = 3.5,
        time_stop_days: int = 60,
        max_pairs: int = 20,
        gross_exposure: float = 1.0,
    ):
        self.sector_groups = sector_groups
        self.formation_months = formation_months
        self.retest_months = retest_months
        self.pvalue_max = pvalue_max
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.time_stop_days = time_stop_days
        self.max_pairs = max_pairs
        self.gross_exposure = gross_exposure

    def required_tickers(self) -> list[str]:
        out: list[str] = []
        for names in self.sector_groups.values():
            out.extend(names)
        return list(dict.fromkeys(out))

    def _formation_dates(self, index: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Fechas de re-test: una cada `retest_months` meses."""
        months = pd.Series(index, index=index).groupby([index.year, index.month]).last()
        dates = list(months.values)
        return [pd.Timestamp(d) for d in dates[:: self.retest_months]]

    def _select_pairs(self, window: pd.DataFrame) -> list[tuple[str, str, float, float]]:
        """Devuelve [(A, B, beta, mean_spread)] cointegrados en la ventana."""
        selected: list[tuple[str, str, float, float]] = []
        for names in self.sector_groups.values():
            valid = [n for n in names if n in window.columns and window[n].notna().sum() > len(window) * 0.9]
            for a, b in combinations(valid, 2):
                sub = window[[a, b]].dropna()
                if len(sub) < 60:
                    continue
                ya, yb = sub[a].values, sub[b].values
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, pval, _ = coint(ya, yb)
                except Exception:  # noqa: BLE001
                    continue
                if pval < self.pvalue_max:
                    beta = self._hedge_ratio(sub[a], sub[b])
                    selected.append((a, b, pval, beta))
        selected.sort(key=lambda x: x[2])  # menor p-value primero
        return [(a, b, beta, np.nan) for a, b, _p, beta in selected[: self.max_pairs]]

    @staticmethod
    def _hedge_ratio(ya: pd.Series, yb: pd.Series) -> float:
        x = sm.add_constant(yb.values)
        model = sm.OLS(ya.values, x).fit()
        return float(model.params[1])

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.required_tickers() if c in prices.columns]
        prices = prices[cols]
        index = prices.index
        target = pd.DataFrame(0.0, index=index, columns=cols)

        formation_days = self.formation_months * 21
        per_pair_gross = self.gross_exposure / max(self.max_pairs, 1)
        leg = per_pair_gross / 2.0  # mitad long, mitad short

        form_dates = self._formation_dates(index)
        active_pairs: list[tuple[str, str, float]] = []  # (A, B, beta)
        # estado por par: dict con position (+1/-1/0), entry_day_idx
        state: dict[tuple[str, str], dict] = {}

        next_form_i = 0
        pos_at = index.get_indexer  # helper

        for i, dt in enumerate(index):
            # ¿Toca re-formar pares?
            while next_form_i < len(form_dates) and dt >= form_dates[next_form_i]:
                window = prices.loc[:dt].iloc[-formation_days:]
                if len(window) >= 60:
                    pairs = self._select_pairs(window)
                    active_pairs = [(a, b, beta) for a, b, beta, _ in pairs]
                    # Resetear estado: cerrar posiciones de pares que ya no existen.
                    new_state: dict[tuple[str, str], dict] = {}
                    for a, b, _ in active_pairs:
                        new_state[(a, b)] = state.get((a, b), {"position": 0, "entry_i": -1})
                    state = new_state
                next_form_i += 1

            if not active_pairs:
                continue

            window = prices.loc[:dt]
            if len(window) < self.zscore_window + 1:
                continue

            for a, b, beta in active_pairs:
                key = (a, b)
                st = state[key]
                sub = window[[a, b]].dropna()
                if len(sub) < self.zscore_window + 1:
                    continue
                spread = sub[a] - beta * sub[b]
                roll = spread.iloc[-self.zscore_window:]
                mu, sd = roll.mean(), roll.std(ddof=0)
                if sd == 0 or np.isnan(sd):
                    continue
                z = (spread.iloc[-1] - mu) / sd

                pos = st["position"]

                if pos == 0:
                    # ¿Abrir?
                    if z >= self.entry_z:
                        # spread alto: A caro vs B → short A, long B
                        st["position"] = -1
                        st["entry_i"] = i
                    elif z <= -self.entry_z:
                        st["position"] = +1
                        st["entry_i"] = i
                else:
                    held_days = i - st["entry_i"]
                    # ¿Cerrar? vuelta a la media, stop por divergencia o por tiempo.
                    revert = (pos == -1 and z <= self.exit_z) or (pos == +1 and z >= self.exit_z)
                    diverge = abs(z) >= self.stop_z
                    timeout = held_days >= self.time_stop_days
                    if revert or diverge or timeout:
                        st["position"] = 0
                        st["entry_i"] = -1

                # Aplicar pesos según posición actual.
                pos = st["position"]
                if pos == -1:      # short A, long B
                    target.loc[dt, a] += -leg
                    target.loc[dt, b] += +leg
                elif pos == +1:    # long A, short B
                    target.loc[dt, a] += +leg
                    target.loc[dt, b] += -leg

        return target
