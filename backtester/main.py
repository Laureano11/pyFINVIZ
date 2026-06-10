"""CLI del backtester de 5 estrategias (plan "151 Trading Strategies").

Uso:
    python -m backtester.main list
    python -m backtester.main run dual_momentum [--start 2010-01-01] [--end 2026-06-09] [--no-tearsheet]
    python -m backtester.main all          # corre las 5 + comparativa + correlaciones
"""
from __future__ import annotations

import argparse

import pandas as pd

from .analysis.compare import ALL_STRATEGIES, run_all
from .analysis.metrics import compute_metrics, metrics_table
from .analysis.report import comparison_html, html_tearsheet
from .runner import run_strategy


def _cmd_list() -> None:
    print("Estrategias disponibles:")
    for s in ALL_STRATEGIES:
        print(f"  - {s}")


def _cmd_run(args: argparse.Namespace) -> None:
    name = args.strategy
    if name not in ALL_STRATEGIES:
        print(f"Estrategia '{name}' desconocida. Opciones: {', '.join(ALL_STRATEGIES)}")
        return

    print(f"\n=== {name} | {args.start or 'config'} → {args.end or 'config'} ===")
    res, metrics, spy = run_strategy(
        name, start=args.start, end=args.end, refresh=args.refresh
    )

    gross = compute_metrics(res.returns_gross, benchmark=spy)
    net = metrics
    table = metrics_table({f"{name}_BRUTO": gross, f"{name}_NETO": net})
    if spy is not None:
        table = metrics_table(
            {
                f"{name}_BRUTO": gross,
                f"{name}_NETO": net,
                "SPY_buyhold": compute_metrics(spy.dropna(), benchmark=spy),
            }
        )
    print(table.to_string())
    print(f"\nturnover anual: {res.annual_turnover:.2f}")

    if not args.no_tearsheet:
        path = html_tearsheet(res.returns_net, name=name, benchmark=spy)
        if path:
            print(f"tearsheet: {path}")


def _cmd_all(args: argparse.Namespace) -> None:
    print(f"\n=== Corriendo las {len(ALL_STRATEGIES)} estrategias ===")
    table, corr, results = run_all(start=args.start, end=args.end, refresh=args.refresh)

    print("\n--- Tabla comparativa (métricas) ---")
    print(table.to_string())

    if not corr.empty:
        print("\n--- Matriz de correlaciones (retornos netos diarios) ---")
        print(corr.round(2).to_string())

    # Tearsheets individuales.
    if not args.no_tearsheet:
        for name, (res, _m, spy) in results.items():
            p = html_tearsheet(res.returns_net, name=name, benchmark=spy)
            if p:
                print(f"tearsheet {name}: {p}")

    out = comparison_html(table, corr if not corr.empty else None)
    print(f"\ncomparativa HTML: {out}")


def _cmd_walkforward(args: argparse.Namespace) -> None:
    from .analysis.sensitivity import walk_forward

    if args.strategy not in ALL_STRATEGIES:
        print(f"Estrategia '{args.strategy}' desconocida.")
        return
    print(f"\n=== Walk-forward: {args.strategy} (in-sample vs out-of-sample) ===")
    table = walk_forward(args.strategy)
    print(table.to_string(index=False))


def _cmd_sensitivity(args: argparse.Namespace) -> None:
    from .analysis.sensitivity import parameter_grid

    if args.strategy not in ALL_STRATEGIES:
        print(f"Estrategia '{args.strategy}' desconocida.")
        return
    # Grids por defecto según el plan §5.6 (lookbacks, umbrales, deciles).
    default_grids = {
        "momentum_xs": {"lookback_months": [3, 6, 9, 12], "quantile": [0.1, 0.2]},
        "dual_momentum": {"lookback_months": [3, 6, 9, 12], "top_n": [2, 3, 4]},
        "trend_following": {"lookback_months": [3, 6, 9, 12]},
        "mean_reversion": {"lookback_days": [3, 5, 10]},
        "pairs": {"entry_z": [1.5, 2.0, 2.5], "stop_z": [3.0, 3.5]},
    }
    grid = default_grids.get(args.strategy, {})
    if not grid:
        print(f"Sin grid por defecto para {args.strategy}.")
        return
    print(f"\n=== Sensibilidad: {args.strategy} | grid {grid} ===")
    table = parameter_grid(args.strategy, grid, start=args.start, end=args.end)
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtester de 5 estrategias.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="listar estrategias")

    p_run = sub.add_parser("run", help="correr una estrategia")
    p_run.add_argument("strategy")
    p_run.add_argument("--start", default=None)
    p_run.add_argument("--end", default=None)
    p_run.add_argument("--refresh", action="store_true", help="re-descargar datos")
    p_run.add_argument("--no-tearsheet", action="store_true")

    p_all = sub.add_parser("all", help="correr todas + comparativa")
    p_all.add_argument("--start", default=None)
    p_all.add_argument("--end", default=None)
    p_all.add_argument("--refresh", action="store_true")
    p_all.add_argument("--no-tearsheet", action="store_true")

    p_wf = sub.add_parser("walkforward", help="in-sample vs out-of-sample (§5.5)")
    p_wf.add_argument("strategy")

    p_sens = sub.add_parser("sensitivity", help="grid de parámetros (§5.6)")
    p_sens.add_argument("strategy")
    p_sens.add_argument("--start", default=None)
    p_sens.add_argument("--end", default=None)

    args = parser.parse_args()
    if args.command == "list":
        _cmd_list()
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "all":
        _cmd_all(args)
    elif args.command == "walkforward":
        _cmd_walkforward(args)
    elif args.command == "sensitivity":
        _cmd_sensitivity(args)


if __name__ == "__main__":
    main()
