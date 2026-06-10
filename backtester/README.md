# Backtester — 5 estrategias de "151 Trading Strategies"

Motor de backtesting vectorizado y 5 estrategias del paper de Kakushadze & Serur,
con métricas estandarizadas, reportes comparables y reglas anti-look-ahead.
Ver `../plan.md` para la especificación completa.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Listar estrategias
python -m backtester.main list

# Correr una estrategia (genera tearsheet HTML en backtester/reports/)
python -m backtester.main run dual_momentum
python -m backtester.main run trend_following --start 2010-01-01 --end 2026-06-09

# Correr las 5 + tabla comparativa + matriz de correlaciones
python -m backtester.main all
```

La primera corrida descarga y cachea los precios en `data/cache/*.parquet`
(yfinance es lento; las siguientes leen del disco). Usar `--refresh` para
re-descargar.

## Las 5 estrategias

| Nombre | Paper § | Tipo | Lado |
|---|---|---|---|
| `dual_momentum` | §4.1.2 | Rotación sectorial doble momentum | long-only |
| `trend_following` | §4.6 | Tendencia multi-activo (inverse-vol) | long-only |
| `momentum_xs` | §3.1 | Momentum cross-sectional 12-1 (deciles) | L/S o long-only |
| `mean_reversion` | §3.9 | Reversión a la media por grupo | dólar-neutral |
| `pairs` | §3.8 | Pares por cointegración (Engle-Granger) | dólar-neutral |

## Arquitectura

```
backtester/
├── config/          # universe.yaml, strategies.yaml, sp500.txt
├── data/loader.py   # yfinance + cache parquet (recorta al rango pedido)
├── strategies/      # base.py + 5 estrategias (contrato: generate_weights → pesos)
├── engine/          # backtest.py (shift anti-look-ahead + costos) + costs.py
├── analysis/        # metrics.py, report.py (quantstats), compare.py
├── runner.py        # orquesta config → estrategia → datos → backtest → métricas
├── main.py          # CLI
└── tests/           # 21 tests (motor, anti-look-ahead, contrato de pesos)
```

## Regla central (anti-look-ahead, plan §4)

```
retorno_portfolio(t) = Σᵢ wᵢ(t-1) · rᵢ(t) - costos(Δw)
```

Toda estrategia produce un `DataFrame` de pesos objetivo calculados con datos
hasta `t`. El **motor** los shiftea un día (no la estrategia). El peso decidido
con el close de `t-1` gana el retorno de `t`. Validado en `tests/test_engine.py`
(buy&hold SPY replica SPY exacto) y en tests `*_no_lookahead`.

## Costos (plan §5)

- Comisión: 10 bps por lado sobre el turnover (`Σ|Δw|`), configurable.
- Shorting: 35 bps anuales sobre la exposición corta (solo dólar-neutral).
- Siempre se reporta **bruto Y neto**: las dólar-neutral generan alfa bruto pero
  típicamente mueren en costos a escala retail (plan §8) — el backtest lo confirma
  con números.

## Tests

```bash
python -m pytest backtester/tests/ -q
```
