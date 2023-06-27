from pathlib import Path
from typing import Any

import pytest

import polars as pl


@pytest.mark.slow()
def test_streaming_out_of_core_unique(
    io_files_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_STREAMING_GROUPBY_SPILL_SIZE", "256")
    df = pl.read_csv(io_files_path / "foods*.csv")
    # this creates 10M rows
    q = df.lazy()
    q = q.join(q, how="cross").select(df.columns).head(10_000)

    # uses out-of-core unique
    df1 = q.join(q.head(1000), how="cross").unique().collect(streaming=True)
    # this ensures the cross join gives equal result but uses the in-memory unique
    df2 = q.join(q.head(1000), how="cross").collect(streaming=True).unique()
    assert df1.shape == df2.shape
    err = capfd.readouterr().err
    assert "OOC groupby started" in err
