# pyFINVIZ

## Stack Tecnologico

- Python 3
- pandas
- numpy
- yfinance
- requests
- python-dotenv
- pandas-ta
- plotly
- dash
- scipy
- statsmodels
- schedule

## Que es este proyecto

pyFINVIZ es una herramienta de analisis tecnico y backtesting para acciones, pensada para usarse desde consola.

El proyecto descarga historicos de mercado, calcula indicadores tecnicos y evalua distintas estrategias de entrada y salida. Tambien puede buscar oportunidades de compra activas en varios tickers al mismo tiempo y, de forma opcional, consultar datos basicos de Financial Modeling Prep para un simbolo puntual.

## Que hace

- Descarga precios historicos con yfinance.
- Calcula indicadores como EMA, MA, MACD, RSI, RSI(2), Williams %R, MFI, Stoch RSI, ATR, ADX e IBS.
- Ejecuta 9 estrategias tecnicas predefinidas.
- Permite analizar una sola accion o una lista de tickers en lote.
- Escanea oportunidades de entrada actuales en base a las estrategias definidas.
- Muestra resumen de trades, ROI, PnL y cantidad de señales.

## Requisitos

- Python 3.10 o superior recomendado.
- Acceso a internet para descargar datos de mercado.
- Opcional: una API key de Financial Modeling Prep si queres activar la consulta extra de perfil y quote.

## Instalacion local

1. Clona el repositorio.

   ```bash
   git clone <URL-del-repositorio>
   cd pyFINVIZ
   ```

2. Crea y activa un entorno virtual.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias.

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta la aplicacion.

   ```bash
   python main.py
   ```

## Uso

Al ejecutar main.py vas a ver un menu con dos modos:

- Analisis de estrategias: corre backtests sobre un ticker o sobre una lista de tickers.
- Busqueda de entrys: busca oportunidades de compra activas en el ultimo dato disponible.

Despues de elegir el modo, el programa te pide la estrategia a evaluar y usa la configuracion del entorno para el resto de los parametros.

## Configuracion por variables de entorno

El proyecto lee variables desde un archivo .env. Las mas utiles son:

- SYMBOL: simbolo usado para la consulta opcional a FMP.
- BACKTEST_SYMBOL: ticker unico para backtesting cuando no se usa una lista.
- BACKTEST_SYMBOLS: lista de tickers separada por comas para analisis en lote.
- BACKTEST_PERIOD: periodo historico a descargar, por ejemplo 10y.
- BACKTEST_INTERVAL: intervalo de velas, por ejemplo 1d.
- ROI_HORIZON_DAYS: cantidad de velas a futuro para medir el resultado.
- MA200_FILTER: filtro respecto de la media de 200 periodos. 0 = none, 1 = above, 2 = below.
- TRADE_CAPITAL_USD: capital teorico por operacion.
- STRATEGY_CHOICE: estrategia por defecto del 1 al 9.
- BATCH_WORKERS: cantidad de workers para procesar tickers en paralelo.
- FMP_API_KEY: api key de Financial Modeling Prep.
- ENABLE_FMP: 1 para activar la consulta a FMP, 0 para desactivarla.

Parametros extra usados por algunas estrategias:

- MACD_RSI_MAX
- MACD_RSI_SELL_MIN
- MFI_MAX
- MFI_SELL_MIN
- RSI2_BUY_LEVEL
- RSI2_SELL_LEVEL
- WILLR_BUY_LEVEL
- WILLR_SELL_LEVEL
- IBS_BUY_LEVEL
- IBS_SELL_LEVEL
- ADX_EMA_MIN_ADX
- STOCHRSI_K_MAX

## Ejemplo de .env

```dotenv
SYMBOL=AAPL
BACKTEST_SYMBOL=AAPL
BACKTEST_SYMBOLS=AAPL,MSFT,NVDA
BACKTEST_PERIOD=10y
BACKTEST_INTERVAL=1d
ROI_HORIZON_DAYS=20
MA200_FILTER=2
TRADE_CAPITAL_USD=1000
STRATEGY_CHOICE=1
BATCH_WORKERS=8
ENABLE_FMP=0
FMP_API_KEY=tu_api_key
```

## Flujo de ejecucion recomendado

1. Crear el entorno virtual.
2. Instalar dependencias.
3. Configurar el archivo .env si queres cambiar simbolos, horizonte o estrategia.
4. Ejecutar python main.py.
5. Elegir el modo de analisis.
6. Revisar el resumen de señales, trades y rendimiento que imprime la terminal.

## Estrategias disponibles

1. Golden Cross
2. MACD + RSI
3. Williams %R
4. IBS
5. IBS + RSI(2)
6. MACD + RSI + MFI
7. Stoch RSI + MACD
8. EMA50/200 + ADX + RSI
9. ADX + EMA direccion

## Notas

- El proyecto esta pensado para correr desde consola, no como web app.
- Si ENABLE_FMP=0, el programa sigue funcionando solo con yfinance.
- Para backtests mas rapidos en lote, sube o baja BATCH_WORKERS segun tu maquina.
