# pyFINVIZ — Contexto del Proyecto

## Descripción general

Script de análisis financiero en Python que extrae datos históricos de precios vía **yfinance** y aplica estrategias técnicas sobre ellos. Tiene dos modos de uso principales:

1. **Análisis de estrategias** — backtesting: prueba el rendimiento histórico de una estrategia sobre uno o varios tickers, calculando ROI, PnL, trades positivos/negativos, señales por año, etc.
2. **Búsqueda de entrys** — scanner: analiza una lista de tickers y detecta cuáles tienen señal de compra activa hoy según cada estrategia disponible.

---

## Estructura del proyecto

```
pyFINVIZ/
├── main.py                  # Entry point, menú de selección de modo
├── pyfinviz/
│   ├── config.py            # Carga de configuración desde .env y env vars
│   ├── data_sources.py      # Descarga de precios (yfinance) y datos FMP
│   ├── engine.py            # Lógica de backtesting y scanner de entrys
│   ├── indicators.py        # Cálculo de indicadores técnicos (RSI, MACD, ATR, etc.)
│   ├── strategies.py        # Definición de las 9 estrategias disponibles
│   ├── reporting.py         # Output formateado con colores por terminal
│   └── sp500_list.py        # Lista de 501 tickers actuales del S&P 500 (2026)
├── .env                     # Variables de configuración (no commiteado sensible)
└── requirements.txt
```

---

## Flujo de datos

```
.env → config.py → engine.py → data_sources.py (yfinance)
                              → indicators.py
                              → strategies.py
                              → reporting.py / print
```

---

## Modo 1: Análisis de estrategias (backtesting)

- Descarga histórico de precios para uno o varios tickers.
- Calcula indicadores técnicos sobre el DataFrame.
- Aplica la estrategia seleccionada para generar señales de entrada y salida.
- Para cada señal de entrada, simula un trade con capital fijo (`TRADE_CAPITAL_USD`) y mide el resultado a `ROI_HORIZON_DAYS` ruedas.
- Soporta SL/TP por ATR en las estrategias que lo definen.
- En modo batch (lista de tickers), usa `ThreadPoolExecutor` para paralelizar.

**Outputs:** tabla de trades individuales coloreados + resumen estadístico (total trades, avg ROI, PnL total, etc.).

---

## Modo 2: Búsqueda de entrys (scanner)

- Toma la lista de tickers (`BACKTEST_SYMBOLS` o S&P 500 completo por default).
- Para cada ticker, descarga el histórico y calcula indicadores.
- Evalúa la última vela de cada estrategia — si hay señal de compra activa, registra la oportunidad.
- Corre en paralelo con `ThreadPoolExecutor`.

**Outputs:** tabla de oportunidades activas con ticker, estrategia, precio de entrada y TP estimado + resumen por estrategia + resumen por ticker.

---

## Estrategias disponibles (1–9)

| Key | Nombre | Descripción |
|-----|--------|-------------|
| 1 | Golden Cross | MA20 cruza por encima de MA50 |
| 2 | MACD + RSI | MACD alcista + RSI < 40 |
| 3 | Williams %R | Cae bajo -80 (sobreventa) |
| 4 | IBS | IBS < 0.20 (close cerca del low) |
| 5 | IBS + RSI(2) | Close > MA200 + RSI2 < 10 + IBS < 0.20 |
| 6 | MACD + RSI + MFI | MACD alcista + RSI < 40 + MFI < 40 |
| 7 | Stoch RSI + MACD | StochRSI K bajo + MACD sobre signal |
| 8 | EMA50/200 + ADX + RSI | Golden cross con filtro ADX > 20, SL/TP por ATR |
| 9 | ADX + EMA dirección | ADX mínimo + Close > EMA14 |

Todas las estrategias soportan filtro MA200 configurable (`ABOVE`, `BELOW`, `NONE`).

---

## Indicadores calculados

`EMA14`, `EMA20`, `EMA50`, `EMA200`, `MA20`, `MA50`, `MA200`, `MACD`, `MACD_SIGNAL`, `RSI(14)`, `RSI(2)`, `Williams %R`, `MFI(14)`, `StochRSI K/D`, `ATR(14)`, `ADX(14)`, `+DI14`, `-DI14`, `IBS`

---

## Configuración (.env)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `BACKTEST_SYMBOLS` | *(vacío → usa SP500)* | Lista de tickers separados por coma para ambos modos |
| `BACKTEST_SYMBOL` | `AAPL` | Ticker único (modo análisis single) |
| `BACKTEST_PERIOD` | `365d` | Período histórico a descargar |
| `BACKTEST_INTERVAL` | `1d` | Temporalidad (`1d`, `1wk`, etc.) |
| `ROI_HORIZON_DAYS` | `20` | Ruedas a medir para el ROI en backtesting |
| `MA200_FILTER` | `1` | `0`=ninguno, `1`=precio sobre MA200, `2`=precio bajo MA200 |
| `TRADE_CAPITAL_USD` | `1000` | Capital por trade en backtesting |
| `BATCH_WORKERS` | `8` | Threads paralelos para batch/scanner |
| `STRATEGY_CHOICE` | *(pregunta)* | Pre-seleccionar estrategia sin prompt interactivo |
| `FMP_API_KEY` | — | API key de Financial Modeling Prep (opcional) |
| `ENABLE_FMP` | `0` | Activar consultas a FMP |

---

## Bugs corregidos

### Bug principal (commit `38b25a2`)
`config.py` hardcodeaba `backtest_symbols = list(SP500_TICKERS)` ignorando la variable de entorno `BACKTEST_SYMBOLS`. Esto rompía el modo 2 porque:
- No había forma de especificar tickers propios.
- Siempre escaneaba los 500+ tickers del S&P 500.
- La lista de SP500 era de 2018, con decenas de empresas deslistadas (ATVI, AET, ALXN, AGN, etc.), generando spam de errores y resultados vacíos.

**Fix aplicado:**
- Restaurado el soporte de `BACKTEST_SYMBOLS` env var: si está definido, lo usa; si no, cae al SP500 completo.
- `sp500_list.py` actualizado a los 501 componentes actuales del S&P 500 (mayo 2026).

---

## Dependencias principales

- `yfinance` — descarga de precios históricos
- `pandas` — manipulación de datos y DataFrames
- `python-dotenv` — carga de `.env`
- `requests` — llamadas HTTP a FMP

---

## Componente 2: `backtester/` — backtester de portafolios (plan.md)

Proyecto independiente del scanner `pyfinviz/`. Implementa las 5 estrategias del
paper "151 Trading Strategies" (Kakushadze & Serur) con un motor de backtesting
vectorizado común. Ver `plan.md` y `backtester/README.md`.

- **Estrategias:** `dual_momentum` (§4.1.2), `trend_following` (§4.6),
  `momentum_xs` (§3.1), `mean_reversion` (§3.9), `pairs` (§3.8, cointegración).
- **Contrato:** cada estrategia produce un DataFrame de pesos objetivo; el motor
  (`engine/backtest.py`) los shiftea 1 día (anti-look-ahead) y aplica costos.
- **Datos:** `data/loader.py` con cache parquet. El cache se baja SIEMPRE en el
  rango canónico amplio (2000-2030); `start`/`end` solo recortan al servir, para
  que un backtest in-sample no corrompa el cache (bug corregido).
- **CLI:** `python -m backtester.main {list|run|all|walkforward|sensitivity}`.
- **Reportes:** tearsheets HTML por estrategia (quantstats) + comparativa +
  matriz de correlaciones, en `backtester/reports/`.
- **Tests:** `python -m pytest backtester/tests/` (31 tests: motor, anti-look-ahead,
  contrato de pesos, recorte de cache).
- **Deps extra:** `statsmodels`, `quantstats`, `matplotlib`, `pyarrow`, `pyyaml`.
