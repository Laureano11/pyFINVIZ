import unittest
from unittest.mock import patch

import pandas as pd

from pyfinviz.data_sources import get_market_history
from pyfinviz.ibkr import normalize_ibkr_bar_size, normalize_ibkr_duration


class IBKRHelpersTest(unittest.TestCase):
    def test_normalize_ibkr_duration(self) -> None:
        self.assertEqual(normalize_ibkr_duration("10y"), "10 Y")
        self.assertEqual(normalize_ibkr_duration("6mo"), "6 M")
        self.assertEqual(normalize_ibkr_duration("30d"), "30 D")

    def test_normalize_ibkr_bar_size(self) -> None:
        self.assertEqual(normalize_ibkr_bar_size("1d"), "1 day")
        self.assertEqual(normalize_ibkr_bar_size("15m"), "15 mins")
        self.assertEqual(normalize_ibkr_bar_size("1wk"), "1 week")


class MarketHistoryFallbackTest(unittest.TestCase):
    def test_falls_back_to_yfinance_when_ibkr_fails(self) -> None:
        expected = pd.DataFrame({"Close": [1.0]})

        with patch("pyfinviz.data_sources.get_ibkr_history", side_effect=RuntimeError("no ibkr")), patch(
            "pyfinviz.data_sources.get_yfinance_history", return_value=expected
        ), patch("builtins.print") as mocked_print:
            result = get_market_history(
                symbol="AAPL",
                period="10y",
                interval="1d",
                source="ibkr",
                fallback_to_yfinance=True,
            )

        self.assertIs(result, expected)
        mocked_print.assert_called()

    def test_raises_when_ibkr_fails_without_fallback(self) -> None:
        with patch("pyfinviz.data_sources.get_ibkr_history", side_effect=RuntimeError("no ibkr")):
            with self.assertRaises(RuntimeError):
                get_market_history(
                    symbol="AAPL",
                    period="10y",
                    interval="1d",
                    source="ibkr",
                    fallback_to_yfinance=False,
                )


if __name__ == "__main__":
    unittest.main()
