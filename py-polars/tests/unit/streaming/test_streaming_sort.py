from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.xdist_group("streaming")


def assert_df_sorted_by(
    df: pl.DataFrame,
    sort_df: pl.DataFrame,
    cols: list[str],
    descending: list[bool] | None = None,
) -> None:
    if descending is None:
        descending = [False] * len(cols)

    # Is sorted by the key columns?
    keycols = sort_df[cols]
    equal = keycols.head(-1) == keycols.tail(-1)

    # Tuple inequality.
    # a0 < b0 || (a0 == b0 && (a1 < b1 || (a1 == b1 && ...))
    # Evaluating in reverse is easiest.
    ordered = equal[cols[-1]]
    for c, desc in zip(cols[::-1], descending[::-1]):
        ordered &= equal[c]
        if desc:
            ordered |= keycols[c].head(-1) > keycols[c].tail(-1)
        else:
            ordered |= keycols[c].head(-1) < keycols[c].tail(-1)

    assert ordered.all()

    # Do all the rows still exist?
    assert Counter(df.rows()) == Counter(sort_df.rows())


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

    result = pl.LazyFrame(data).sort("foo", "baz").collect(streaming=True)

    expected = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": ["c", "b", "a"],
            "baz": [
                datetime(2023, 5, 1, 14, 45),
                datetime(2023, 5, 1, 13, 45),
                datetime(2023, 5, 1, 15, 45),
            ],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.write_disk()
@pytest.mark.slow()
def test_ooc_sort(tmp_path: Path, monkeypatch: Any) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")

    s = pl.arange(0, 100_000, eager=True).rename("idx")

    df = s.shuffle().to_frame()

    for descending in [True, False]:
        out = (
            df.lazy().sort("idx", descending=descending).collect(streaming=True)
        ).to_series()

        assert_series_equal(out, s.sort(descending=descending))


@pytest.mark.debug()
@pytest.mark.write_disk()
@pytest.mark.parametrize("spill_source", [True, False])
def test_streaming_sort(
    tmp_path: Path, monkeypatch: Any, capfd: Any, spill_source: bool
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    if spill_source:
        monkeypatch.setenv("POLARS_SPILL_SORT_PARTITIONS", "1")
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
    if spill_source:
        assert "PARTITIONED FORCE SPILLED" in err


@pytest.mark.write_disk()
@pytest.mark.parametrize("spill_source", [True, False])
def test_out_of_core_sort_9503(
    tmp_path: Path, monkeypatch: Any, spill_source: bool
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    if spill_source:
        monkeypatch.setenv("POLARS_SPILL_SORT_PARTITIONS", "1")
    np.random.seed(0)

    num_rows = 100_000
    num_columns = 2
    num_tables = 10

    # ensure we create many chunks
    # this will ensure we create more files
    # and that creates contention while dumping
    df = pl.concat(
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
    )
    lf = df.lazy()

    result = lf.sort(df.columns).collect(streaming=True)

    assert result.shape == (1_000_000, 2)
    assert result["column_0"].flags["SORTED_ASC"]
    assert result.head(20).to_dict(as_series=False) == {
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


@pytest.mark.write_disk()
@pytest.mark.slow()
def test_streaming_sort_multiple_columns(
    str_ints_df: pl.DataFrame, tmp_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
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
    df = q.collect()
    assert_df_sorted_by(df, q.sort(sort_by).collect(streaming=True), sort_by)
    assert_df_sorted_by(df, q.sort(sort_by).collect(streaming=False), sort_by)


def test_streaming_sort_fixed_reverse() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 1, 2, 4, 1, 7],
            "b": [1, 2, 2, 1, 2, 4, 8, 7],
        }
    )
    descending = [True, False]
    q = df.lazy().sort(by=["a", "b"], descending=descending)

    assert_df_sorted_by(
        df, q.collect(streaming=True), ["a", "b"], descending=descending
    )
    assert_df_sorted_by(
        df, q.collect(streaming=False), ["a", "b"], descending=descending
    )


def test_reverse_variable_sort_13573() -> None:
    df = pl.DataFrame(
        {
            "a": ["one", "two", "three"],
            "b": ["four", "five", "six"],
        }
    ).lazy()
    assert df.sort("a", "b", descending=[True, False]).collect(streaming=True).to_dict(
        as_series=False
    ) == {"a": ["two", "three", "one"], "b": ["five", "six", "four"]}


def test_nulls_last_streaming_sort() -> None:
    assert pl.LazyFrame({"x": [1, None]}).sort("x", nulls_last=True).collect(
        streaming=True
    ).to_dict(as_series=False) == {"x": [1, None]}


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("nulls_last", [True, False])
def test_sort_descending_nulls_last(descending: bool, nulls_last: bool) -> None:
    df = pl.DataFrame({"x": [1, 3, None, 2, None], "y": [1, 3, 0, 2, 0]})

    null_sentinel = 100 if descending ^ nulls_last else -100
    ref_x = [1, 3, None, 2, None]
    ref_x.sort(key=lambda k: null_sentinel if k is None else k, reverse=descending)
    ref_y = [1, 3, 0, 2, 0]
    ref_y.sort(key=lambda k: null_sentinel if k == 0 else k, reverse=descending)

    assert_frame_equal(
        df.lazy()
        .sort("x", descending=descending, nulls_last=nulls_last)
        .collect(streaming=True),
        pl.DataFrame({"x": ref_x, "y": ref_y}),
    )

    assert_frame_equal(
        df.lazy()
        .sort(["x", "y"], descending=descending, nulls_last=nulls_last)
        .collect(streaming=True),
        pl.DataFrame({"x": ref_x, "y": ref_y}),
    )
