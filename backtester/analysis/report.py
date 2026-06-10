from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def html_tearsheet(
    returns: pd.Series,
    *,
    name: str,
    benchmark: pd.Series | None = None,
) -> Path | None:
    """Genera un tearsheet HTML con quantstats. Devuelve la ruta o None si falla."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{name}.html"
    try:
        import quantstats as qs

        rets = returns.dropna()
        rets.index = pd.to_datetime(rets.index)
        bench = None
        if benchmark is not None:
            bench = benchmark.reindex(rets.index).dropna()
            rets = rets.reindex(bench.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(
                rets,
                benchmark=bench,
                output=str(out),
                title=name,
                download_filename=str(out),
            )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[report] tearsheet '{name}' falló: {exc}")
        return None


def comparison_html(table: pd.DataFrame, correlations: pd.DataFrame | None = None) -> Path:
    """Tabla comparativa de métricas + matriz de correlaciones en un HTML."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "comparison.html"
    parts = [
        "<html><head><meta charset='utf-8'><title>Comparativa de estrategias</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;}"
        "table{border-collapse:collapse;margin-bottom:2rem;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;}"
        "th{background:#222;color:#fff;}tr:nth-child(even){background:#f5f5f5;}"
        "caption{font-weight:bold;font-size:1.1rem;margin-bottom:.5rem;text-align:left;}"
        "</style></head><body>",
        "<h1>Backtester — comparativa de 5 estrategias</h1>",
        "<table><caption>Métricas (neto de costos salvo indicado)</caption>",
        table.to_html(border=0),
        "</table>",
    ]
    if correlations is not None:
        parts += [
            "<table><caption>Correlación de retornos diarios netos</caption>",
            correlations.round(2).to_html(border=0),
            "</table>",
        ]
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
