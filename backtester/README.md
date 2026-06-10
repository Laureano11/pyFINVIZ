# Guía completa del Backtester

## 1. Qué es y cómo funciona (el modelo mental)

El sistema prueba **estrategias de portafolio** sobre datos históricos. La idea central es una sola, y todo gira alrededor de ella:

> **Toda estrategia produce una matriz de pesos** (`fechas × tickers`). El motor multiplica esos pesos por los retornos del día siguiente y resta costos. Eso es el backtest.

```
retorno_del_día(t) = Σᵢ peso_de_ayer(t-1, i) · retorno_hoy(t, i) − costos
```

El detalle **innegociable** es ese `t-1`: el peso se decide con datos de ayer y gana el retorno de hoy. Esto evita el "look-ahead" (hacer trampa mirando el futuro). El motor hace ese desfase automáticamente — las estrategias no se preocupan por eso.

### Flujo de datos

```
config/*.yaml  ──►  runner.py  ──►  loader.py ──► cache parquet (descarga 1 vez)
                       │
                       ▼
              strategies/*.py  (generate_weights → matriz de pesos)
                       │
                       ▼
              engine/backtest.py  (shift t-1 + costos)  ──►  retornos netos
                       │
                       ▼
              analysis/  (métricas, tearsheet HTML, comparativa)
```

### Las 5 estrategias

| Comando | Qué hace | Tipo |
|---|---|---|
| `dual_momentum` | Compra los 3 sectores SPDR más fuertes, pero solo si el mercado está sobre su MA200; si no, va a bonos/oro | Long-only |
| `trend_following` | De ~10 ETFs de distintas clases, se queda con los que suben y están sobre MA200, ponderados por baja volatilidad | Long-only |
| `momentum_xs` | Del S&P 500, compra el decil de mayor momentum 12-1 (long-only por config) | Long-only |
| `mean_reversion` | Dentro de cada sector, short lo que subió de más / long lo que cayó de más | Dólar-neutral |
| `pairs` | Encuentra pares cointegrados y opera su spread cuando se desvía ±2σ | Dólar-neutral |

---

## 2. Deploy / instalación

El sistema es **100% local, gratis, sin API keys**. No hay nada que desplegar a un servidor — corre en tu máquina.

### Pasos de instalación (desde cero, en otra máquina)

```bash
# 1. Clonar y entrar
git clone <repo> && cd pyFINVIZ
git checkout backtester-5-estrategias   # la rama con el backtester

# 2. Crear entorno e instalar
python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac (en Windows: .venv\Scripts\activate)
pip install -r backtester/requirements.txt

# 3. Verificar que todo funciona
python -m pytest backtester/tests/ -q     # deben pasar 36 tests
python -m backtester.main list            # lista las 5 estrategias
```

**En tu máquina actual ya está todo instalado.** Solo necesitás activar el venv:

```bash
cd /home/lauri11/Documentos/pyFINVIZ
source .venv/bin/activate
```

> **Nota sobre la primera corrida:** la primera vez que tocás una estrategia, descarga los precios de yfinance y los cachea en `backtester/data/cache/*.parquet`. El S&P 500 (momentum_xs) son ~500 tickers y tarda **~10-15 min la primera vez**. Después lee del disco (segundos). Los ETFs (dual_momentum, trend_following) son rápidos. En tu máquina el cache **ya está poblado**, así que no vas a re-descargar.

---

## 3. Cómo usarlo — los 6 comandos

Todos se invocan con `python -m backtester.main <comando>`.

### `list` — ver estrategias disponibles
```bash
python -m backtester.main list
```

### `run` — correr UNA estrategia
```bash
python -m backtester.main run dual_momentum
python -m backtester.main run momentum_xs --start 2015-01-01 --end 2024-12-31
python -m backtester.main run pairs --no-tearsheet   # sin generar el HTML
```
Imprime una tabla **bruto vs neto vs SPY** y genera un tearsheet HTML en `backtester/reports/<estrategia>.html`.

**Flags:** `--start` / `--end` (rango de fechas), `--refresh` (re-descargar datos ignorando cache), `--no-tearsheet` (no generar el HTML, más rápido).

### `all` — correr las 5 + comparativa + correlaciones
```bash
python -m backtester.main all --start 2012-01-01 --end 2026-06-09
```
Es el comando estrella. Produce la tabla comparativa, la matriz de correlaciones entre las 5, todos los tearsheets, y un `reports/comparison.html`.

### `walkforward` — validar que no esté sobreajustada (§5.5)
```bash
python -m backtester.main walkforward dual_momentum
```
Compara el rendimiento **in-sample** (2005-2015) vs **out-of-sample** (2016-2026). Si la estrategia colapsa fuera de muestra, está overfitteada.

### `sensitivity` — grid de parámetros (§5.6)
```bash
python -m backtester.main sensitivity dual_momentum --start 2012-01-01 --end 2026-06-09
```
Corre la estrategia con muchas combinaciones de parámetros. Si solo funciona con UNA, es sospechosa.

---

## 4. Cómo evaluar una estrategia (lo más importante)

Esta es la parte donde la mayoría se engaña. El sistema te da las herramientas para no hacerlo. Hay **4 preguntas** que tenés que responder, en orden:

### Pregunta 1: ¿Le gana a comprar SPY y olvidarse?

```bash
python -m backtester.main run dual_momentum --start 2012-01-01 --end 2026-06-09 --no-tearsheet
```

```
=== dual_momentum | 2012-01-01 → 2026-06-09 ===
                      CAGR    Vol  Sharpe  Sortino  MaxDD  Calmar  PctPosMonths  BetaVsBench  TurnoverAnnual
dual_momentum_BRUTO   9.84  14.84    0.71     0.93 -22.99    0.43         55.17         0.41             NaN
dual_momentum_NETO    8.87  14.84    0.65     0.85 -23.81    0.37         54.60         0.41             8.85
SPY_buyhold          14.94  16.58    0.92     1.13 -33.72    0.44         70.11         1.00             NaN
```

Mirá esta tabla — así se lee:

- **Mirá siempre la fila `_NETO`**, no la `_BRUTO`. La bruta ignora costos y miente.
- **Sharpe** (retorno ajustado por riesgo): el número que más importa. Acá dual_momentum neto = 0.65 vs SPY = 0.92. **SPY le gana en Sharpe.**
- **PERO** mirá `MaxDD` (la peor caída): dual_momentum -23.8% vs SPY -33.7%. Y `BetaVsBench` 0.41 (se mueve menos que el mercado). → Es una estrategia **defensiva**: rinde menos pero sufre menos en las crisis.

La regla del plan (§6): **si una estrategia no le gana a SPY ajustada por riesgo y neta de costos, no se opera** — por más linda que sea la teoría.

### Pregunta 2: ¿El costo de transacción la mata?

Compará las filas `_BRUTO` y `_NETO`. Si el Sharpe se desploma de bruto a neto, la estrategia rota demasiado. Ejemplo extremo:

```bash
python -m backtester.main run mean_reversion --start 2012-01-01 --end 2026-06-09 --no-tearsheet
```

```
mean_reversion_BRUTO   1.24   4.95    0.27     0.40 -16.30    0.08         52.87         0.05             NaN
mean_reversion_NETO   -6.02   5.03   -1.21    -1.76 -60.26   -0.10         25.86         0.05            72.50
SPY_buyhold           14.94  16.58    0.92     1.13 -33.72    0.44         70.11         1.00             NaN
```

Caso de libro: `mean_reversion` pasa de **Sharpe +0.27 bruto a −1.21 neto**. ¿Por qué? `TurnoverAnnual = 72.5` — rota su cartera entera 72 veces al año. A 10 bps por operación, los costos se la comen viva. El plan (§8) lo predijo: *"las dólar-neutral probablemente mueran en costos a escala retail"*. El backtest lo confirma con números, que es exactamente para lo que sirve.

> El error `404 symbol K` que puede aparecer es cosmético — es Kellanova (renombrada), que yfinance ya no encuentra. El sistema la omite y sigue.

### Pregunta 3: ¿Generaliza fuera de muestra? (walk-forward)

```bash
python -m backtester.main walkforward dual_momentum
```

```
=== Walk-forward: dual_momentum (in-sample vs out-of-sample) ===
       period       from         to  CAGR_%  Sharpe  MaxDD_%
    in_sample 2005-01-01 2015-12-31   11.45   0.735   -26.59
out_of_sample 2016-01-01 2026-06-09    7.56   0.547   -23.81
```

El rendimiento baja de in-sample (Sharpe 0.74) a out-of-sample (0.55) — **es normal y esperable** que baje un poco. Lo que querés ver es que **no colapse**: sigue positivo y con Sharpe > 0.5 en datos que la estrategia "nunca vio". Eso indica que captura un efecto real, no ruido. Si el OOS fuera negativo o cerca de 0, sería sobreajuste.

### Pregunta 4: ¿Es robusta a sus parámetros? (sensibilidad)

```bash
python -m backtester.main sensitivity dual_momentum --start 2012-01-01 --end 2026-06-09
```

```
=== Sensibilidad: dual_momentum | grid {'lookback_months': [3, 6, 9, 12], 'top_n': [2, 3, 4]} ===
 lookback_months  top_n  CAGR_%  Sharpe  MaxDD_%  Turnover
               3      2    9.60   0.657   -29.13     12.47
               3      3    9.93   0.707   -28.42     11.02
               3      4    8.82   0.659   -29.61      9.67
               6      2    7.03   0.511   -30.06     10.26
               6      3    8.87   0.647   -23.81      8.85
               6      4    9.16   0.682   -23.32      7.93
               9      2    7.98   0.576   -30.75      8.80
               9      3    9.19   0.684   -29.38      7.32
               9      4    9.70   0.737   -28.44      6.55
              12      2    9.89   0.682   -23.50      8.32
              12      3    8.86   0.650   -22.90      7.65
              12      4   10.53   0.773   -22.54      6.41
```

Mirá la columna `Sharpe`: va de **0.51 a 0.77** en las 12 combinaciones. **Nunca colapsa.** Eso es robustez: la estrategia funciona con lookback de 3, 6, 9 o 12 meses y comprando 2, 3 o 4 sectores.

Contraejemplo de lo que sería malo: si 11 combinaciones dieran Sharpe ~0 y solo una diera 1.5, esa estrategia estaría **overfitteada** a ese parámetro mágico y habría que descartarla (regla §5.6 del plan).

### El veredicto de evaluación, resumido

Una estrategia "pasa" si:
1. ✅ Su **Sharpe neto** es competitivo vs SPY (o gana en MaxDD si es defensiva).
2. ✅ El Sharpe **no se desploma** de bruto a neto (turnover manejable).
3. ✅ El **out-of-sample** no colapsa.
4. ✅ Es **robusta** en el grid de sensibilidad.

Por eso, de las 5: **`momentum_xs` (long-only), `dual_momentum` y `trend_following` superan o compiten con SPY**, y `mean_reversion` / `pairs` quedan como **diversificadores** (no ganan solas, pero su correlación ~0 con el resto aporta al combo).

> **Por qué momentum_xs es long-only y no L/S.** El paper define momentum como long/short dólar-neutral (long ganadores, short perdedores). Medido en 2012-2026, esa versión L/S da Sharpe ~0.18 — el factor momentum L/S se degradó globalmente desde ~2010, y el lado short en un bull market es un lastre. La **misma señal** operada solo long (comprar el decil ganador, sin shorts) da **Sharpe 1.09 y CAGR +24%**, ganándole a SPY. Por eso el default es `long_short: false`. Si querés estudiar el factor puro, poné `long_short: true` en `config/strategies.yaml`.

---

## 5. El combo — el objetivo final (§3.20/§6)

El plan dice que el objetivo final no es una estrategia sola, sino **combinarlas**: cuando las estrategias están poco correlacionadas, el portafolio combinado tiene mejor Sharpe y mucho menor drawdown que cualquiera de sus partes.

```bash
# Combina las 3 que superan a SPY (equal-weight, recomendado)
python -m backtester.main combo --survivors --start 2012-01-01 --end 2026-06-09

# Combina las 5 (incluye los diversificadores)
python -m backtester.main combo --start 2012-01-01 --end 2026-06-09

# Risk-parity (inverse-vol) en vez de equal-weight
python -m backtester.main combo --survivors --method risk_parity
```

**Resultado del combo de las 3 sobrevivientes (equal-weight, 2012-2026):**

```
                     CAGR    Vol  Sharpe  Sortino  MaxDD  Calmar  BetaVsBench
COMBO_equal_weight  12.78  12.66    1.01     1.26 -22.70    0.56         0.59
SPY_buyhold         14.94  16.58    0.92     1.13 -33.72    0.44         1.00
```

El combo **le gana a SPY en las 3 dimensiones que importan**: Sharpe (1.01 vs 0.92), drawdown (-22.7% vs -33.7%) y Calmar (0.56 vs 0.44). Captura el 85% del retorno de SPY con dos tercios de su riesgo de cola. **Eso es el "combo alfa" del §3.20** — el verdadero entregable del proyecto, no las estrategias individuales.

> **Sobre `--method`:** `equal_weight` (default) reparte parejo. `risk_parity` pondera por inverse-vol — útil en general, pero ojo: si incluís las perdedoras (`mean_reversion`, `pairs`), risk-parity les da MÁS peso por su baja volatilidad, hundiendo el combo. Por eso para el combo de las 5 conviene equal-weight, y para el de sobrevivientes cualquiera de los dos anda bien.

---

## 6. Leer los reportes HTML

```
backtester/reports/buyhold_spy.html           1723 KB
backtester/reports/comparison.html               3 KB
backtester/reports/dual_momentum.html         1576 KB
backtester/reports/mean_reversion.html        1534 KB
backtester/reports/momentum_xs.html           1578 KB
backtester/reports/pairs.html                 1475 KB
backtester/reports/trend_following.html       1589 KB
```

- **`comparison.html`** — tabla comparativa de las 5 + matriz de correlaciones (vista de pájaro).
- **`<estrategia>.html`** — tearsheet completo de quantstats: curva de equity, drawdowns, retornos mensuales/anuales, distribuciones, rolling Sharpe, etc.

Para abrirlos:
```bash
xdg-open backtester/reports/comparison.html        # Linux
# o simplemente arrastrá el archivo al navegador
```

La **matriz de correlaciones** del `comparison.html` es clave para el objetivo final del plan (el "combo alfa" §3.20): aunque las dólar-neutral pierdan solas, tienen correlación **casi cero** con las defensivas. Combinar estrategias descorrelacionadas reduce el riesgo total del conjunto.

---

## 7. Personalizar — los archivos de config

Toda la parametrización está en YAML, **sin tocar código**:

```yaml
# strategies.yaml (parámetros)

# Modelo de costos común (plan §5).
costs:
  commission_bps: 10        # por lado, sobre el turnover (|Δw|)
  short_borrow_bps_annual: 35   # costo anual de shorting sobre exposición corta

# §4.1.2 — Rotación sectorial doble momentum.
dual_momentum:
  lookback_months: 6        # retorno acumulado 6m (el paper sugiere 6-12)
  rebalance: "M"
  top_n: 3                  # comprar top 3 sectores
  abs_ma_days: 200          # filtro SPY > MA(200)
```

- **`config/strategies.yaml`** — todos los parámetros: lookbacks, umbrales, deciles, **y el modelo de costos** (`commission_bps`, `short_borrow_bps_annual`). Querés probar costos más altos/bajos, o momentum_xs long-only en vez de L/S (`long_short: false`)? Lo cambiás acá.
- **`config/universe.yaml`** — qué tickers usa cada estrategia (sectores, ETFs, refugios, fechas de descarga).
- **`config/sp500.txt`** — la lista del S&P 500 (un ticker por línea) para las cross-sectional.

Ejemplo: si querés ver cuánto cambian las dólar-neutral con costos institucionales (1 bps en vez de 10), editás `commission_bps: 1` y re-corrés — vas a ver que dejan de morir.

---

## 8. Resumen de funcionalidades

| Funcionalidad | Comando / archivo |
|---|---|
| Listar estrategias | `main.py list` |
| Backtest individual + tearsheet | `main.py run <estrategia>` |
| Comparativa de las 5 + correlaciones | `main.py all` |
| Validación out-of-sample | `main.py walkforward <estrategia>` |
| Robustez de parámetros | `main.py sensitivity <estrategia>` |
| Combo de las 5 (§3.20/§6) | `main.py combo [--method risk_parity\|equal_weight]` |
| Bruto vs neto de costos | en cada `run` (filas `_BRUTO`/`_NETO`) |
| Métricas (Sharpe, Sortino, MaxDD, Calmar, beta, turnover…) | `analysis/metrics.py` |
| Tearsheets HTML | `reports/*.html` |
| Cambiar parámetros sin tocar código | `config/strategies.yaml` |
| Cambiar universo de tickers | `config/universe.yaml` |
| Verificar integridad del sistema | `pytest backtester/tests/` (36 tests) |

---

## Sobre "deploy" — una aclaración importante

Esto es un **backtester de investigación**, no un bot de trading en vivo. El plan lo deja explícito (§7): la ejecución en vivo es la "Fase 5, futura, solo si la Fase 4 da señal". El sistema te dice **qué estrategias valdría la pena operar**, pero no manda órdenes a ningún broker todavía.

Si en algún momento querés llevar a vivo las que superan a SPY (`momentum_xs` long-only, `dual_momentum`, `trend_following`), el plan sugiere el camino: paper trading 3-6 meses primero, y como rebalancean **una vez al mes**, son operables incluso a mano. Ahí sí entraría una integración con broker (IBKR, o CEDEARs por broker local) — que es justamente lo que el otro componente del repo (`pyfinviz/ibkr.py`) ya explora por separado.

> Ojo con `momentum_xs` long-only: aunque tiene el mejor Sharpe, su **MaxDD es -38%** (mayor que SPY) y opera ~45 acciones del S&P 500 que rotan cada mes — más trabajo operativo que las 3-10 posiciones de dual_momentum/trend_following. Mejor Sharpe no es gratis.
