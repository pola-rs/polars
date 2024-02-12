from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.write_disk()
@pytest.mark.slow()
def test_streaming_out_of_core_unique(
    io_files_path: Path, tmp_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
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

    # TODO: Re-enable this check when this issue is fixed: https://github.com/pola-rs/polars/issues/10466
    _ = capfd.readouterr().err
    # assert "OOC group_by started" in err


def test_streaming_unique(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    df = pl.DataFrame({"a": [1, 2, 2, 2], "b": [3, 4, 4, 4], "c": [5, 6, 7, 7]})
    q = df.lazy().unique(subset=["a", "c"], maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))

    q = df.lazy().unique(subset=["b", "c"], maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))

    q = df.lazy().unique(subset=None, maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))
    (_, err) = capfd.readouterr()
    assert "df -> re-project-sink -> sort_multiple" in err
