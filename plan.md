# Plan técnico — Backtester de 5 estrategias (basado en "151 Trading Strategies", Kakushadze & Serur)

## 1. Resumen

- **Objetivo:** implementar en Python las 5 estrategias seleccionadas del paper, con un motor de backtesting común, métricas estandarizadas y reportes comparables.
- **Alcance:** backtesting histórico con datos diarios gratuitos (yfinance). No incluye ejecución en vivo (eso es una fase posterior).
- **Criterio de selección de estrategias:** el paper no rankea rentabilidad — es descriptivo. La selección se basa en evidencia académica citada en el propio paper + viabilidad con datos OHLCV gratis + diversidad de estilos (momentum, mean reversion, trend following, stat arb).

## 2. Las 5 estrategias y su especificación

### 2.1 Precio-momentum cross-sectional (§3.1)
- **Lógica:** ranking mensual de acciones por retorno acumulado de 12 meses salteando el último (12-1, como indica el paper para evitar la reversión de corto plazo). Long decil superior, short decil inferior, dólar-neutral. Variante long-only para comparar.
- **Universo:** S&P 500 (lista actual como aproximación; ver riesgo de survivorship bias en §7).
- **Rebalanceo:** mensual. Período de tenencia: 1 mes.
- **Ponderación:** módulo-uniforme (1/2N_L y -1/2N_C) y variante inverse-vol (w ∝ 1/σᵢ) como propone el paper.
- **Evidencia:** Jegadeesh & Titman (1993); el factor con más literatura detrás.

### 2.2 Rotación sectorial con doble momentum (§4.1.2)
- **Lógica:** cada mes, rankear los 11 ETFs sectoriales SPDR (XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLB, XLU, XLRE, XLC) por retorno acumulado 6-12 meses. Comprar los top 3 **solo si** SPY > MA(200) (momentum absoluto). Si no, rotar a un activo refugio (TLT o GLD), como indica el paper.
- **Rebalanceo:** mensual. Long-only, sin apalancamiento ni shorts.
- **Evidencia:** Antonacci (2014, 2017), citado por el paper. El filtro de régimen es lo que históricamente recortó los drawdowns de 2008 y 2020.

### 2.3 Trading de pares (§3.8, mejorado con cointegración)
- **Lógica:** el paper usa desviación de la correlación histórica; lo implementamos con el estándar moderno: test de cointegración Engle-Granger sobre pares del mismo sector. Cuando el spread (residuo de la regresión) supera ±2σ de su media rolling → short la cara / long la barata, dólar-neutral (Q_A·P_A = -Q_B·P_B como en las Ec. 290-291). Cierre cuando el spread vuelve a la media o stop por divergencia (±3.5σ) o por tiempo (60 días).
- **Selección de pares:** ventana de formación 12 meses, re-test trimestral, operar solo pares con p-value < 0.05.
- **Evidencia:** Gatev, Goetzmann & Rouwenhorst (2006), citado en el paper.

### 2.4 Reversión a la media – grupo único (§3.9)
- **Lógica:** dentro de un grupo de acciones correlacionadas (un sector), calcular el retorno neto de la media R̃ᵢ = Rᵢ - R̄ sobre los últimos d días (d=5 por defecto). Posiciones en dólares Dᵢ ∝ -R̃ᵢ (vender lo que subió de más, comprar lo que cayó de más), normalizando con γ para que Σ|Dᵢ| = I (Ec. 295-298 del paper). Dólar-neutral por construcción.
- **Rebalanceo:** semanal (diario es más fiel al espíritu de la estrategia pero los costos de transacción la matan a escala retail — se backtestean ambos para verlo con números).
- **Evidencia:** es la familia de stat arb de los propios autores (Kakushadze 2015).

### 2.5 Seguimiento de tendencia multi-activo (§4.6)
- **Lógica:** universo de ~10 ETFs de clases de activos distintas (SPY, EFA, EEM, TLT, IEF, GLD, DBC, VNQ, etc.). Cada mes: filtrar los que tengan retorno acumulado 6-12m positivo **y** precio > MA(200) (el doble filtro que describe el paper). A los sobrevivientes, ponderación inverse-volatility (Ec. 372: wᵢ ∝ 1/σᵢ), normalizada a Σwᵢ = 1. Lo filtrado queda en cash (o SHY).
- **Rebalanceo:** mensual. Long-only.
- **Evidencia:** Faber (2007), Moskowitz, Ooi & Pedersen (2012), citados en el paper.

## 3. Decisiones de stack

| Componente | Elección | Por qué | Alternativa descartada y motivo |
|---|---|---|---|
| Lenguaje | Python 3.11+ | Ecosistema cuant estándar | — |
| Datos | `yfinance` | Gratis, precios ajustados por splits/dividendos (requisito explícito del paper), suficiente para daily | APIs pagas (Polygon, EODHD): innecesarias para validar; se agregan si algo pasa a vivo |
| Manipulación | `pandas` + `numpy` | Estándar, todo el backtesting es vectorizable a frecuencia mensual/semanal | `polars`: más rápido pero el cuello de botella acá no es performance |
| Estadística | `statsmodels` | `coint()` (Engle-Granger) y OLS para pares; ADF tests | Implementar ADF a mano: sin sentido |
| Motor de backtest | **Custom vectorizado** (~300 líneas) | Las 5 estrategias son portafolios con rebalanceo periódico — un motor de matrices de pesos × matriz de retornos es trivial, transparente y auditable | `backtrader`: event-driven, lento y verboso para portafolios cross-sectional. `zipline`: semi-abandonado. `vectorbt`: excelente para señales single-asset, incómodo para portafolios dólar-neutral con N variable |
| Métricas/reportes | `quantstats` | Tearsheets HTML completos (Sharpe, Sortino, max DD, CAGR, rolling stats) en una línea | Calcular métricas a mano: reinventar la rueda |
| Gráficos extra | `matplotlib` | Para gráficos específicos (spreads de pares, exposición) | `plotly`: overkill |
| Cache de datos | `parquet` local (`pyarrow`) | yfinance es lento y rate-limitea; descargar una vez, leer del disco siempre | SQLite: más piezas para cero beneficio acá |
| Config | `YAML` (`pyyaml`) | Parámetros de cada estrategia fuera del código → grid de parámetros fácil | Hardcodear: impide el análisis de sensibilidad |

**Instalación:**
```bash
pip install yfinance pandas numpy statsmodels quantstats matplotlib pyarrow pyyaml
```

## 4. Arquitectura

```
backtester/
├── config/
│   ├── universe.yaml          # tickers por estrategia
│   └── strategies.yaml        # parámetros (lookbacks, deciles, umbrales σ, etc.)
├── data/
│   ├── loader.py              # descarga yfinance + cache parquet
│   └── cache/                 # parquets por ticker
├── strategies/
│   ├── base.py                # interfaz: generate_weights(prices) -> DataFrame de pesos
│   ├── momentum_xs.py         # §3.1
│   ├── dual_momentum.py       # §4.1.2
│   ├── pairs.py               # §3.8
│   ├── mean_reversion.py      # §3.9
│   └── trend_following.py     # §4.6
├── engine/
│   ├── backtest.py            # pesos × retornos, costos, slippage, equity curve
│   └── costs.py               # modelo de costos de transacción
├── analysis/
│   ├── metrics.py             # wrapper quantstats + métricas propias
│   └── report.py              # tearsheet HTML por estrategia + tabla comparativa
├── main.py                    # CLI: correr una estrategia o todas
└── requirements.txt
```

**Contrato central:** toda estrategia produce un `DataFrame` de pesos objetivo (fechas × tickers, sum |w| ≤ 1). El motor hace lo mismo para las 5:

```
retorno_portfolio(t) = Σᵢ wᵢ(t-1) · rᵢ(t) - costos(Δw)
```

Los pesos se calculan con datos hasta t-1 y se aplican al retorno de t. **Esto es la regla anti-look-ahead y es innegociable.**

## 5. Reglas de backtesting (donde la mayoría se miente a sí misma)

1. **Sin look-ahead:** señal con close de t-1, ejecución al open/close de t. Shift explícito de pesos en el motor, no en cada estrategia.
2. **Costos de transacción:** 10 bps por lado por defecto (configurable). Las estrategias dólar-neutral (§3.1, §3.8, §3.9) rotan mucho — sin costos parecen máquinas de plata; con costos es donde se separan las viables de las académicas. Reportar siempre bruto Y neto.
3. **Costo de shorting:** 25-50 bps anuales sobre el lado corto para las dólar-neutral. Retail en Argentina ni siquiera puede shortear fácil — la versión long-only es la realista; la L/S se backtestea para entender el factor.
4. **Survivorship bias:** usar la lista actual del S&P 500 sobrefavorece al momentum (~1-2% anual de inflación del resultado). Mitigación barata: documentarlo y descontar mentalmente. Mitigación correcta (fase 2): listas históricas de constituyentes.
5. **Walk-forward / out-of-sample:** parámetros elegidos en 2005-2015, validados en 2016-2026. Nunca reportar solo in-sample.
6. **Sensibilidad de parámetros:** grid sobre lookbacks (3/6/9/12m), umbrales (1.5/2/2.5σ), deciles vs quintiles. Si la estrategia solo funciona con UNA combinación, está overfitteada y se descarta.

## 6. Métricas de evaluación (por estrategia, bruto y neto)

CAGR, volatilidad anualizada, Sharpe, Sortino, max drawdown, Calmar, % meses positivos, turnover anual, beta vs SPY, correlación entre las 5 estrategias (el objetivo final es un combo — el "combo alfa" del §3.20 del paper — y ahí la correlación importa más que el Sharpe individual).

**Benchmark:** SPY buy & hold. Si una estrategia no le gana ajustada por riesgo neta de costos, no se opera, por más linda que sea la teoría.

## 7. Fases

### Fase 0 — Infraestructura (2-3 días con Claude Code)
- `loader.py` + cache parquet, motor de backtest con shift y costos, wrapper quantstats.
- Test del motor: estrategia dummy buy & hold SPY debe replicar el retorno de SPY exacto. Si no, el motor está roto.

### Fase 1 — Estrategias simples primero (2-3 días)
- §4.1.2 (dual momentum) y §4.6 (trend following): son las más simples, long-only, validan el pipeline completo.

### Fase 2 — Cross-sectional (3-4 días)
- §3.1 (momentum) y §3.9 (mean reversion): requieren manejar universo de ~500 tickers, NaNs, delistings.

### Fase 3 — Pares (2-3 días)
- §3.8: la más compleja (formación de pares, cointegración, gestión de posiciones abiertas con stops).

### Fase 4 — Análisis comparativo
- Tabla comparativa, matriz de correlaciones, sensibilidad de parámetros, walk-forward. Decisión: cuáles sobreviven netas de costos.

### Fase 5 (futura, solo si Fase 4 da señal) — Paper trading
- Ejecución simulada en vivo 3-6 meses antes de poner un peso. Recién ahí evaluar broker/API (IBKR para US, o CEDEARs vía broker local con ejecución manual mensual — dual momentum y trend following rebalancean 1 vez al mes, son operables a mano).

## 8. Riesgos y honestidad

- **Las dólar-neutral probablemente mueran en costos a escala retail.** Momentum L/S y mean reversion diaria funcionan institucionalmente con costos de 1-2 bps. A 10 bps + sin acceso barato a shorts, lo esperable es que solo sobrevivan las long-only (§4.1.2, §4.6) y quizás pares con baja frecuencia. El backtest sirve justamente para confirmarlo con números en vez de creerlo.
- **Momentum tiene crashes:** 2009 fue -73% para el factor WML. El filtro de momentum absoluto del §4.1.2 mitiga, no elimina.
- **yfinance no es contractual:** puede romperse o cambiar. Mitigación: capa `loader.py` aislada — cambiar de proveedor toca un solo archivo.
- **Decisión reversible:** todo el stack es local y gratis. Lo único irreversible es el tiempo: timebox de ~2 semanas para Fases 0-4, consistente con tu regla de cortar rápido.

## 9. Checklist de arranque

- [ ] Crear repo `backtester/` con la estructura de §4
- [ ] `pip install -r requirements.txt`
- [ ] Implementar `loader.py` y descargar/cachear: S&P 500 actual, 11 SPDR sectoriales, 10 ETFs multi-activo, SPY/TLT/GLD/SHY (histórico 2000-2026)
- [ ] Motor de backtest + test dummy buy & hold
- [ ] Estrategia 1: dual momentum (§4.1.2) end-to-end con tearsheet
- [ ] Iterar Fases 2-4
