from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def _ann_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    total = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def _ann_vol(returns: pd.Series) -> float:
    return float(returns.dropna().std(ddof=0) * np.sqrt(TRADING_DAYS))


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series, rf: float = 0.0) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _calmar(returns: pd.Series) -> float:
    mdd = _max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(_ann_return(returns) / abs(mdd))


def _pct_positive_months(returns: pd.Series) -> float:
    monthly = (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean() * 100.0)


def beta_vs_benchmark(returns: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    var = cov[1, 1]
    if var == 0:
        return 0.0
    return float(cov[0, 1] / var)


def compute_metrics(
    returns: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    turnover_annual: float | None = None,
) -> dict[str, float]:
    """Diccionario de métricas estándar (plan §6)."""
    m: dict[str, float] = {
        "CAGR": _ann_return(returns),
        "Vol": _ann_vol(returns),
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "MaxDD": _max_drawdown(returns),
        "Calmar": _calmar(returns),
        "PctPosMonths": _pct_positive_months(returns),
    }
    if benchmark is not None:
        m["BetaVsBench"] = beta_vs_benchmark(returns, benchmark)
    if turnover_annual is not None:
        m["TurnoverAnnual"] = float(turnover_annual)
    return m


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convierte {nombre: metrics} en una tabla comparativa formateada."""
    df = pd.DataFrame(results).T
    # _pct_positive_months ya devuelve en %; el resto son fracciones.
    pct_cols = ["CAGR", "Vol", "MaxDD"]
    for c in pct_cols:
        if c in df.columns:
            df[c] = (df[c] * 100).round(2)
    if "PctPosMonths" in df.columns:
        df["PctPosMonths"] = df["PctPosMonths"].round(2)
    for c in ["Sharpe", "Sortino", "Calmar", "BetaVsBench", "TurnoverAnnual"]:
        if c in df.columns:
            df[c] = df[c].round(2)
    return df
