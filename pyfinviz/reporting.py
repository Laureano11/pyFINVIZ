import pandas as pd

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"


def print_colored_trades(trades_df: pd.DataFrame, limit: int = 20) -> None:
    if trades_df.empty:
        print("No hay trades para mostrar.")
        return

    print(
        "signal_date  entry_date   exit_date    entry_price  exit_price  holding_days  "
        "roi_pct   pnl_usd  final_usd"
    )
    for _, row in trades_df.tail(limit).iterrows():
        signal_date = row["signal_date"].strftime("%Y-%m-%d")
        entry_date = row["entry_date"].strftime("%Y-%m-%d")
        exit_date = row["exit_date"].strftime("%Y-%m-%d")
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        holding_days = int(row["holding_days"])
        roi_pct = float(row["roi_pct"])
        pnl_usd = float(row["pnl_usd"])
        final_usd = float(row["final_usd"])

        if roi_pct > 0:
            color = ANSI_GREEN
        elif roi_pct < 0:
            color = ANSI_RED
        else:
            color = ""

        line = (
            f"{signal_date:<12} {entry_date:<12} {exit_date:<12} "
            f"{entry_price:>11.2f} {exit_price:>10.2f} {holding_days:>13} "
            f"{roi_pct:>8.2f}% {pnl_usd:>9.2f} {final_usd:>10.2f}"
        )
        print(f"{color}{line}{ANSI_RESET}" if color else line)
