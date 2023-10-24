from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

pytestmark = pytest.mark.xdist_group("streaming")


def test_streaming_sort_multiple_columns_logical_types() -> None:
    data = {
        "foo": [3, 2, 1],
        "bar": ["a", "b", "c"],
        "baz": [
            datetime(2023, 5, 1, 15, 45),
            datetime(2023, 5, 1, 13, 45),
            datetime(2023, 5, 1, 14, 45),
        ],
    }
    assert pl.DataFrame(data).lazy().sort("foo", "baz").collect(streaming=True).to_dict(
        False
    ) == {
        "foo": [1, 2, 3],
        "bar": ["c", "b", "a"],
        "baz": [
            datetime(2023, 5, 1, 14, 45),
            datetime(2023, 5, 1, 13, 45),
            datetime(2023, 5, 1, 15, 45),
        ],
    }


@pytest.mark.write_disk()
@pytest.mark.slow()
def test_ooc_sort(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    s = pl.arange(0, 100_000, eager=True).rename("idx")

    df = s.shuffle().to_frame()

    for descending in [True, False]:
        out = (
            df.lazy().sort("idx", descending=descending).collect(streaming=True)
        ).to_series()

        assert_series_equal(out, s.sort(descending=descending))


@pytest.mark.write_disk()
def test_streaming_sort(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    # this creates a lot of duplicate partitions and triggers: #7568
    assert (
        pl.Series(np.random.randint(0, 100, 100))
        .to_frame("s")
        .lazy()
        .sort("s")
        .collect(streaming=True)["s"]
        .is_sorted()
    )
    (_, err) = capfd.readouterr()
    assert "df -> sort" in err


@pytest.mark.write_disk()
def test_out_of_core_sort_9503(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    np.random.seed(0)

    num_rows = 1_00_000
    num_columns = 2
    num_tables = 10

    # ensure we create many chunks
    # this will ensure we create more files
    # and that creates contention while dumping
    q = pl.concat(
        [
            pl.DataFrame(
                [
                    pl.Series(np.random.randint(0, 10000, size=num_rows))
                    for _ in range(num_columns)
                ]
            )
            for _ in range(num_tables)
        ],
        rechunk=False,
    ).lazy()
    q = q.sort(q.columns)
    df = q.collect(streaming=True)
    assert df.shape == (1_000_000, 2)
    assert df["column_0"].flags["SORTED_ASC"]
    assert df.head(20).to_dict(False) == {
        "column_0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "column_1": [
            242,
            245,
            588,
            618,
            732,
            902,
            925,
            945,
            1009,
            1161,
            1352,
            1365,
            1451,
            1581,
            1778,
            1836,
            1976,
            2091,
            2120,
            2124,
        ],
    }


@pytest.mark.skip(
    reason="This test is unreliable - it fails intermittently in our CI"
    " with 'OSError: No such file or directory (os error 2)'."
)
@pytest.mark.write_disk()
@pytest.mark.slow()
def test_streaming_sort_multiple_columns(
    str_ints_df: pl.DataFrame, monkeypatch: Any, capfd: Any
) -> None:
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    df = str_ints_df

    out = df.lazy().sort(["strs", "vals"]).collect(streaming=True)
    assert_frame_equal(out, out.sort(["strs", "vals"]))
    err = capfd.readouterr().err
    assert "OOC sort forced" in err
    assert "RUN STREAMING PIPELINE" in err
    assert "df -> sort_multiple" in err
    assert out.columns == ["vals", "strs"]


def test_streaming_sort_sorted_flag() -> None:
    # empty
    q = pl.LazyFrame(
        schema={
            "store_id": pl.UInt16,
            "item_id": pl.UInt32,
            "timestamp": pl.Datetime,
        }
    ).sort("timestamp")

    assert q.collect(streaming=True)["timestamp"].flags["SORTED_ASC"]


@pytest.mark.parametrize(
    ("sort_by"),
    [
        ["fats_g", "category"],
        ["fats_g", "category", "calories"],
        ["fats_g", "category", "calories", "sugars_g"],
    ],
)
def test_streaming_sort_varying_order_and_dtypes(
    io_files_path: Path, sort_by: list[str]
) -> None:
    q = pl.scan_parquet(io_files_path / "foods*.parquet")
    q = q.sort(sort_by)
    assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))
