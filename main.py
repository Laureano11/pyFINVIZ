
import requests
from dotenv import load_dotenv
from pyfinviz.config import load_runtime_config
from pyfinviz.data_sources import get_fmp_profile, get_fmp_quote
from pyfinviz.engine import analyze_strategy_after_n_days, analyze_symbols_batch, scan_entry_opportunities
from pyfinviz.reporting import print_colored_trades
from pyfinviz.strategies import STRATEGIES


def choose_main_mode() -> str:
	print("\nMenu principal:")
	print("1 - Analisis de estrategias")
	print("2 - Busqueda de entrys")

	try:
		selected = input("Elegi opcion (default 1): ").strip() or "1"
	except EOFError:
		selected = "1"

	if selected not in {"1", "2"}:
		print("Opcion invalida. Se usa 1 - Analisis de estrategias.")
		return "1"
	return selected


def run_strategy_analysis() -> None:
	cfg = load_runtime_config(ask_strategy=True)
	strategy = STRATEGIES[cfg.strategy_key]

	if cfg.backtest_symbols:
		print(
			f"\n--- Estrategia: {strategy.name} | Lote ({len(cfg.backtest_symbols)} tickers) | "
			f"Fuente: {cfg.market_data_source} | Filtro MA200: {cfg.ma200_filter_mode} | Temporalidad: {cfg.interval} | ROI a {cfg.horizon_days} ruedas | "
			f"Capital por trade: ${cfg.trade_capital_usd:.2f} | Workers: {cfg.batch_workers} ---"
		)
		summary_df = analyze_symbols_batch(
			symbols=cfg.backtest_symbols,
			period=cfg.period,
			interval=cfg.interval,
			market_data_source=cfg.market_data_source,
			market_data_fallback_to_yfinance=cfg.market_data_fallback_to_yfinance,
			ibkr_host=cfg.ibkr_host,
			ibkr_port=cfg.ibkr_port,
			ibkr_client_id=cfg.ibkr_client_id,
			ibkr_use_rth=cfg.ibkr_use_rth,
			ibkr_what_to_show=cfg.ibkr_what_to_show,
			horizon_days=cfg.horizon_days,
			ma200_filter_mode=cfg.ma200_filter_mode,
			trade_capital_usd=cfg.trade_capital_usd,
			strategy_key=cfg.strategy_key,
			strategy_params=cfg.strategy_params,
			batch_workers=cfg.batch_workers,
		)
		if summary_df.empty:
			print("No se pudieron generar resultados para la lista de tickers.")
		else:
			print("\nResumen por ticker (ordenado por total_pnl_usd):")
			print(summary_df.to_string(index=False))
			print(
				"\nTotales lote:",
				{
					"tickers": int(len(summary_df)),
					"signals": int(summary_df["total_signals"].sum()),
					"avg_signals_per_year": round(float(summary_df["avg_signals_per_year"].mean()), 2),
					"trades": int(summary_df["total_trades"].sum()),
					"total_avg_pnl_per_year_usd": round(float(summary_df["avg_pnl_per_year_usd"].sum()), 2),
					"total_pnl_usd": round(float(summary_df["total_pnl_usd"].sum()), 2),
					"avg_roi_pct": round(float(summary_df["avg_roi_pct"].mean()), 2),
				},
			)
	else:
		print(
			f"\n--- Estrategia: {strategy.name} en {cfg.backtest_symbol} | "
			f"Fuente: {cfg.market_data_source} | Filtro MA200: {cfg.ma200_filter_mode} | Temporalidad: {cfg.interval} | ROI a {cfg.horizon_days} ruedas | "
			f"Capital por trade: ${cfg.trade_capital_usd:.2f} ---"
		)
		trades_df, stats = analyze_strategy_after_n_days(
			symbol=cfg.backtest_symbol,
			period=cfg.period,
			interval=cfg.interval,
			market_data_source=cfg.market_data_source,
			market_data_fallback_to_yfinance=cfg.market_data_fallback_to_yfinance,
			ibkr_host=cfg.ibkr_host,
			ibkr_port=cfg.ibkr_port,
			ibkr_client_id=cfg.ibkr_client_id,
			ibkr_use_rth=cfg.ibkr_use_rth,
			ibkr_what_to_show=cfg.ibkr_what_to_show,
			horizon_days=cfg.horizon_days,
			ma200_filter_mode=cfg.ma200_filter_mode,
			trade_capital_usd=cfg.trade_capital_usd,
			strategy_key=cfg.strategy_key,
			strategy_params=cfg.strategy_params,
		)
		if trades_df.empty:
			print("No se encontraron operaciones para esta regla en el periodo.")
		else:
			print(
				{
					"total_signals": int(stats["total_signals"]),
					"avg_signals_per_year": round(stats["avg_signals_per_year"], 2),
					"total_trades": int(stats["total_trades"]),
					"positive_trades": int(stats["positive_trades"]),
					"negative_trades": int(stats["negative_trades"]),
					"neutral_trades": int(stats["neutral_trades"]),
					"avg_roi_pct": round(stats["avg_roi_pct"], 2),
					"total_invested_usd": round(stats["total_invested_usd"], 2),
					"total_final_usd": round(stats["total_final_usd"], 2),
					"total_pnl_usd": round(stats["total_pnl_usd"], 2),
					"avg_pnl_per_year_usd": round(stats["avg_pnl_per_year_usd"], 2),
					"avg_pnl_usd": round(stats["avg_pnl_usd"], 2),
				},
			)
			print("\nTrades (verde=positivo, rojo=negativo):")
			print_colored_trades(trades_df, limit=20)

	if not cfg.enable_fmp:
		print("\nFMP desactivado para testing (ENABLE_FMP=0).")
		print("Activalo cuando quieras con ENABLE_FMP=1 en .env")
		return

	if not cfg.fmp_api_key:
		print("\nFalta FMP_API_KEY en .env. Ejemplo: FMP_API_KEY=tu_api_key")
		return

	print(f"\n--- FMP ({cfg.symbol}) ---")
	try:
		profile = get_fmp_profile(symbol=cfg.symbol, api_key=cfg.fmp_api_key)
		quote = get_fmp_quote(symbol=cfg.symbol, api_key=cfg.fmp_api_key)
	except requests.HTTPError as exc:
		status = exc.response.status_code if exc.response is not None else "unknown"
		if status == 403:
			print(
				"FMP devolvio 403 (Forbidden). Tu API key no tiene acceso a ese endpoint o plan."
			)
			print("Revisa tu plan o prueba con otro endpoint en FMP Dashboard.")
			return
		print(f"Error HTTP al consultar FMP: {status}")
		return

	if profile:
		print(
			"Perfil:",
			{
				"companyName": profile.get("companyName"),
				"sector": profile.get("sector"),
				"industry": profile.get("industry"),
				"exchange": profile.get("exchangeShortName"),
			},
		)
	else:
		print("No se pudo obtener profile desde FMP.")

	if quote:
		print(
			"Quote:",
			{
				"price": quote.get("price"),
				"change": quote.get("change"),
				"changesPercentage": quote.get("changesPercentage"),
				"volume": quote.get("volume"),
			},
		)
	else:
		print("No se pudo obtener quote desde FMP.")


def run_entry_scanner() -> None:
	cfg = load_runtime_config(ask_strategy=False)
	symbols = cfg.backtest_symbols if cfg.backtest_symbols else [cfg.backtest_symbol]

	print(
		f"\n--- Busqueda de entrys | {len(symbols)} tickers | "
		f"Filtro MA200: {cfg.ma200_filter_mode} | Temporalidad: {cfg.interval} | Estrategias: 1,2,3,8,9 | Workers: {cfg.batch_workers} ---"
	)

	opportunities_df = scan_entry_opportunities(
		symbols=symbols,
		period=cfg.period,
		interval=cfg.interval,
		market_data_source=cfg.market_data_source,
		market_data_fallback_to_yfinance=cfg.market_data_fallback_to_yfinance,
		ibkr_host=cfg.ibkr_host,
		ibkr_port=cfg.ibkr_port,
		ibkr_client_id=cfg.ibkr_client_id,
		ibkr_use_rth=cfg.ibkr_use_rth,
		ibkr_what_to_show=cfg.ibkr_what_to_show,
		ma200_filter_mode=cfg.ma200_filter_mode,
		strategy_params=cfg.strategy_params,
		batch_workers=cfg.batch_workers,
	)

	if opportunities_df.empty:
		print("No hay oportunidades de compra activas hoy en los tickers/estrategias evaluados.")
		return

	opportunities_df = opportunities_df.copy()
	opportunities_df["tp_price"] = opportunities_df["tp_price"].apply(
		lambda v: "-" if v is None else f"{float(v):.4f}"
	)

	print("\nOportunidades de compra actuales:")
	print(opportunities_df.to_string(index=False))

	summary = opportunities_df.groupby("strategy_key").size().reset_index(name="signals")
	summary["strategy_name"] = summary["strategy_key"].map(lambda k: STRATEGIES[str(k)].name)
	summary = summary[["strategy_key", "strategy_name", "signals"]].sort_values(by="signals", ascending=False)

	print("\nResumen por estrategia:")
	print(summary.to_string(index=False))

	by_symbol = (
		opportunities_df.groupby("symbol")
		.agg(
			signals=("strategy_key", "size"),
			strategies=("strategy_name", lambda x: ", ".join(sorted(set(x)))),
			entry_price=("entry_price", "first"),
			tp_strategies=("tp_price", lambda x: int((x != "-").sum())),
		)
		.reset_index()
		.sort_values(by="signals", ascending=False)
	)
	print("\nResumen por ticker:")
	print(by_symbol.to_string(index=False))


def main() -> None:
	load_dotenv()
	mode = choose_main_mode()

	if mode == "2":
		run_entry_scanner()
	else:
		run_strategy_analysis()


if __name__ == "__main__":
	main()
