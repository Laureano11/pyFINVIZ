"""Tests del loader: el recorte por sub-rango no debe corromper el cache amplio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.data import loader


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """Redirige el cache a un tmp dir y mockea la descarga de yfinance."""
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)

    full_idx = pd.date_range("2005-01-03", "2026-06-08", freq="B")
    full = pd.DataFrame(
        {
            "Open": 100.0, "High": 101.0, "Low": 99.0,
            "Close": np.linspace(100, 300, len(full_idx)),
            "Volume": 1_000_000,
        },
        index=full_idx,
    )

    def fake_download(ticker, start, end):
        # Simula yfinance: devuelve el rango pedido recortado del histórico completo.
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        return full.loc[(full.index >= lo) & (full.index <= hi)].copy()

    monkeypatch.setattr(loader, "_download_one", fake_download)
    return full


def test_subrange_query_does_not_shrink_cache(fake_cache, tmp_path):
    """Pedir un sub-rango (2005-2015) debe cachear el rango canónico completo,
    no solo el sub-rango — regresión del bug que recortaba SPY a 2005-2015."""
    # Primera consulta: sub-rango angosto.
    df1 = loader.load_ticker("FAKE", "2005-01-01", "2015-12-31")
    assert df1.index.max().year == 2015  # lo servido respeta el recorte

    # El cache en disco, en cambio, debe cubrir hasta el presente (canónico).
    cached = pd.read_parquet(tmp_path / "FAKE.parquet")
    assert cached.index.max().year >= 2026

    # Segunda consulta pidiendo más allá de 2015: debe servirse del cache amplio
    # sin re-descargar incompleto.
    df2 = loader.load_ticker("FAKE", "2016-01-01", "2026-06-08")
    assert not df2.empty
    assert df2.index.min().year == 2016
    assert df2.index.max().year >= 2026


def test_slice_respects_query_range(fake_cache, tmp_path):
    df = loader.load_ticker("FAKE2", "2010-01-01", "2012-12-31")
    assert df.index.min() >= pd.Timestamp("2010-01-01")
    assert df.index.max() <= pd.Timestamp("2012-12-31")


def test_load_prices_dedups_and_aligns(fake_cache):
    px = loader.load_prices(["FAKE", "FAKE", "FAKE3"], "2010-01-01", "2011-12-31")
    # dedup: FAKE aparece una sola vez como columna.
    assert list(px.columns).count("FAKE") == 1
    assert "FAKE3" in px.columns
