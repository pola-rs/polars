from __future__ import annotations

import base64
import io
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import ComputeError, SchemaError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import ParallelStrategy
    from tests.conftest import PlMonkeyPatch


@pytest.fixture
def parquet_file_path(io_files_path: Path) -> Path:
    return io_files_path / "small.parquet"


@pytest.fixture
def foods_parquet_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.parquet"


def test_scan_parquet(parquet_file_path: Path) -> None:
    df = pl.scan_parquet(parquet_file_path)
    assert df.collect().shape == (4, 3)


def test_scan_parquet_local_with_async(
    plmonkeypatch: PlMonkeyPatch, foods_parquet_path: Path
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    pl.scan_parquet(foods_parquet_path.relative_to(Path.cwd())).head(1).collect()


def test_row_index(foods_parquet_path: Path) -> None:
    df = pl.read_parquet(foods_parquet_path, row_index_name="row_index")
    assert df["row_index"].to_list() == list(range(27))

    df = (
        pl.scan_parquet(foods_parquet_path, row_index_name="row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_parquet(foods_parquet_path, row_index_name="row_index")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_row_index_len_16543(foods_parquet_path: Path) -> None:
    q = pl.scan_parquet(foods_parquet_path).with_row_index()
    assert q.select(pl.all()).select(pl.len()).collect().item() == 27


@pytest.mark.write_disk
def test_categorical_parquet_statistics(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame(
        {
            "book": [
                "bookA",
                "bookA",
                "bookB",
                "bookA",
                "bookA",
                "bookC",
                "bookC",
                "bookC",
            ],
            "transaction_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "user": ["bob", "bob", "bob", "tim", "lucy", "lucy", "lucy", "lucy"],
        }
    ).with_columns(pl.col("book").cast(pl.Categorical))

    file_path = tmp_path / "books.parquet"
    df.write_parquet(file_path, statistics=True)

    parallel_options: list[ParallelStrategy] = [
        "auto",
        "columns",
        "row_groups",
        "none",
    ]
    for par in parallel_options:
        df = (
            pl.scan_parquet(file_path, parallel=par)
            .filter(pl.col("book") == "bookA")
            .collect()
        )
    assert df.shape == (4, 3)


@pytest.mark.write_disk
def test_parquet_eq_stats(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "stats.parquet"

    df1 = pd.DataFrame({"a": [None, 1, None, 2, 3, 3, 4, 4, 5, 5]})
    df1.to_parquet(file_path, engine="pyarrow")
    df = pl.scan_parquet(file_path).filter(pl.col("a") == 4).collect()
    assert df["a"].to_list() == [4.0, 4.0]

    assert (
        pl.scan_parquet(file_path).filter(pl.col("a") == 2).select(pl.col("a").sum())
    ).collect()[0, "a"] == 2.0

    assert pl.scan_parquet(file_path).filter(pl.col("a") == 5).collect().shape == (
        2,
        1,
    )


@pytest.mark.write_disk
def test_parquet_is_in_stats(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "stats.parquet"

    df1 = pd.DataFrame({"a": [None, 1, None, 2, 3, 3, 4, 4, 5, 5]})
    df1.to_parquet(file_path, engine="pyarrow")
    df = pl.scan_parquet(file_path).filter(pl.col("a").is_in([5])).collect()
    assert df["a"].to_list() == [5.0, 5.0]

    assert (
        pl.scan_parquet(file_path)
        .filter(pl.col("a").is_in([5]))
        .select(pl.col("a").sum())
    ).collect()[0, "a"] == 10.0

    assert (
        pl.scan_parquet(file_path)
        .filter(pl.col("a").is_in([1, 2, 3]))
        .select(pl.col("a").sum())
    ).collect()[0, "a"] == 9.0

    assert (
        pl.scan_parquet(file_path)
        .filter(pl.col("a").is_in([1, 2, 3]))
        .select(pl.col("a").sum())
    ).collect()[0, "a"] == 9.0

    assert (
        pl.scan_parquet(file_path)
        .filter(pl.col("a").is_in([5]))
        .select(pl.col("a").sum())
    ).collect()[0, "a"] == 10.0

    assert pl.scan_parquet(file_path).filter(
        pl.col("a").is_in([1, 2, 3, 4, 5])
    ).collect().shape == (8, 1)


@pytest.mark.write_disk
def test_parquet_stats(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "binary_stats.parquet"

    df1 = pd.DataFrame({"a": [None, 1, None, 2, 3, 3, 4, 4, 5, 5]})
    df1.to_parquet(file_path, engine="pyarrow")
    df = (
        pl.scan_parquet(file_path)
        .filter(pl.col("a").is_not_null() & (pl.col("a") > 4))
        .collect()
    )
    assert df["a"].to_list() == [5.0, 5.0]

    assert (
        pl.scan_parquet(file_path).filter(pl.col("a") > 4).select(pl.col("a").sum())
    ).collect()[0, "a"] == 10.0

    assert (
        pl.scan_parquet(file_path).filter(pl.col("a") < 4).select(pl.col("a").sum())
    ).collect()[0, "a"] == 9.0

    assert (
        pl.scan_parquet(file_path).filter(pl.col("a") < 4).select(pl.col("a").sum())
    ).collect()[0, "a"] == 9.0

    assert (
        pl.scan_parquet(file_path).filter(pl.col("a") > 4).select(pl.col("a").sum())
    ).collect()[0, "a"] == 10.0
    assert pl.scan_parquet(file_path).filter(
        (pl.col("a") * 10) > 5.0
    ).collect().shape == (8, 1)


def test_row_index_schema_parquet(parquet_file_path: Path) -> None:
    assert (
        pl.scan_parquet(str(parquet_file_path), row_index_name="id")
        .select(["id", "b"])
        .collect()
    ).dtypes == [pl.get_index_type(), pl.String]


@pytest.mark.may_fail_cloud  # reason: inspects logs
@pytest.mark.write_disk
def test_parquet_is_in_statistics(
    plmonkeypatch: PlMonkeyPatch, capfd: Any, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    df = pl.DataFrame({"idx": pl.arange(0, 100, eager=True)}).with_columns(
        (pl.col("idx") // 25).alias("part")
    )
    df = pl.concat(df.partition_by("part", as_dict=False), rechunk=False)
    assert df.n_chunks("all") == [4, 4]

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False)

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False)

    for pred in [
        pl.col("idx").is_in([150, 200, 300]),
        pl.col("idx").is_in([5, 250, 350]),
    ]:
        result = pl.scan_parquet(file_path).filter(pred).collect()
        assert_frame_equal(result, df.filter(pred))

    captured = capfd.readouterr().err
    assert "Predicate pushdown: reading 1 / 1 row groups" in captured
    assert "Predicate pushdown: reading 0 / 1 row groups" in captured


@pytest.mark.may_fail_cloud  # reason: inspects logs
@pytest.mark.write_disk
def test_parquet_statistics(
    plmonkeypatch: PlMonkeyPatch, capfd: Any, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    df = pl.DataFrame({"idx": pl.arange(0, 100, eager=True)}).with_columns(
        (pl.col("idx") // 25).alias("part")
    )
    df = pl.concat(df.partition_by("part", as_dict=False), rechunk=False)
    assert df.n_chunks("all") == [4, 4]

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False, row_group_size=50)

    for pred in [
        pl.col("idx") < 50,
        pl.col("idx") > 50,
        pl.col("idx").null_count() != 0,
        pl.col("idx").null_count() == 0,
        pl.col("idx").min() == pl.col("part").null_count(),
    ]:
        result = pl.scan_parquet(file_path).filter(pred).collect()
        assert_frame_equal(result, df.filter(pred))

    captured = capfd.readouterr().err

    assert "Predicate pushdown: reading 1 / 2 row groups" in captured


@pytest.mark.write_disk
def test_categorical(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame(
        [
            pl.Series("name", ["Bob", "Alice", "Bob"], pl.Categorical),
            pl.Series("amount", [100, 200, 300]),
        ]
    )

    file_path = tmp_path / "categorical.parquet"
    df.write_parquet(file_path)

    result = (
        pl.scan_parquet(file_path)
        .group_by("name")
        .agg(pl.col("amount").sum())
        .collect()
        .sort("name")
    )
    expected = pl.DataFrame(
        {"name": ["Alice", "Bob"], "amount": [200, 400]},
        schema_overrides={"name": pl.Categorical},
    )
    assert_frame_equal(result, expected)


def test_glob_n_rows(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.parquet"
    df = pl.scan_parquet(file_path, n_rows=40).collect()

    # 27 rows from foods1.parquet and 13 from foods2.parquet
    assert df.shape == (40, 4)

    # take first and last rows
    assert df[[0, 39]].to_dict(as_series=False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }


@pytest.mark.write_disk
def test_parquet_statistics_filter_9925(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "codes.parquet"
    df = pl.DataFrame({"code": [300964, 300972, 500_000, 26]})
    df.write_parquet(file_path, statistics=True)

    q = pl.scan_parquet(file_path).filter(
        (pl.col("code").floordiv(100_000)).is_in([0, 3])
    )
    assert q.collect().to_dict(as_series=False) == {"code": [300964, 300972, 26]}


@pytest.mark.write_disk
def test_parquet_statistics_filter_11069(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "foo.parquet"
    pl.DataFrame({"x": [1, None]}).write_parquet(file_path, statistics=False)

    result = pl.scan_parquet(file_path).filter(pl.col("x").is_null()).collect()
    expected = {"x": [None]}
    assert result.to_dict(as_series=False) == expected


def test_parquet_list_arg(io_files_path: Path) -> None:
    first = io_files_path / "foods1.parquet"
    second = io_files_path / "foods2.parquet"

    df = pl.scan_parquet(source=[first, second]).collect()
    assert df.shape == (54, 4)
    assert df.row(-1) == ("seafood", 194, 12.0, 1)
    assert df.row(0) == ("vegetables", 45, 0.5, 2)


@pytest.mark.write_disk
def test_parquet_many_row_groups_12297(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "foo.parquet"
    df = pl.DataFrame({"x": range(100)})
    df.write_parquet(file_path, row_group_size=5, use_pyarrow=True)
    assert_frame_equal(pl.scan_parquet(file_path).collect(), df)


@pytest.mark.write_disk
def test_row_index_empty_file(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "test.parquet"
    df = pl.DataFrame({"a": []}, schema={"a": pl.Float32})
    df.write_parquet(file_path)
    result = pl.scan_parquet(file_path).with_row_index("idx").collect()
    assert result.schema == OrderedDict([("idx", pl.UInt32), ("a", pl.Float32)])


@pytest.mark.write_disk
def test_io_struct_async_12500(tmp_path: Path) -> None:
    file_path = tmp_path / "test.parquet"
    pl.DataFrame(
        [
            pl.Series("c1", [{"a": "foo", "b": "bar"}], dtype=pl.Struct),
            pl.Series("c2", [18]),
        ]
    ).write_parquet(file_path)
    assert pl.scan_parquet(file_path).select("c1").collect().to_dict(
        as_series=False
    ) == {"c1": [{"a": "foo", "b": "bar"}]}


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_different_schema(tmp_path: Path, streaming: bool) -> None:
    # Schema is different but the projected columns are same dtype.
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    a = pl.DataFrame({"a": [1.0], "b": "a"})

    b = pl.DataFrame({"a": [1], "b": "a"})

    a.write_parquet(f1)
    b.write_parquet(f2)
    assert pl.scan_parquet([f1, f2]).select("b").collect(
        engine="streaming" if streaming else "in-memory"
    ).columns == ["b"]


@pytest.mark.write_disk
def test_nested_slice_12480(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    df = pl.select(pl.lit(1).repeat_by(10_000).explode().cast(pl.List(pl.Int32)))

    df.write_parquet(path, use_pyarrow=True, pyarrow_options={"data_page_size": 1})

    assert pl.scan_parquet(path).slice(0, 1).collect().height == 1


@pytest.mark.write_disk
def test_scan_deadlock_rayon_spawn_from_async_15172(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "1")
    path = tmp_path / "data.parquet"

    df = pl.Series("x", [1]).to_frame()
    df.write_parquet(path)

    results = [pl.DataFrame()]

    def scan_collect() -> None:
        results[0] = pl.collect_all([pl.scan_parquet(path)])[0]

    # Make sure we don't sit there hanging forever on the broken case
    t = Thread(target=scan_collect, daemon=True)
    t.start()
    t.join(5)

    assert results[0].equals(df)


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_schema_mismatch_panic_17067(tmp_path: Path, streaming: bool) -> None:
    pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).write_parquet(tmp_path / "1.parquet")
    pl.DataFrame({"c": [1, 2, 3], "d": [4, 5, 6]}).write_parquet(tmp_path / "2.parquet")

    if streaming:
        with pytest.raises(pl.exceptions.SchemaError):
            pl.scan_parquet(tmp_path).collect(engine="streaming")
    else:
        with pytest.raises(pl.exceptions.SchemaError):
            pl.scan_parquet(tmp_path).collect(engine="in-memory")


@pytest.mark.write_disk
def test_predicate_push_down_categorical_17744(tmp_path: Path) -> None:
    path = tmp_path / "1"

    df = pl.DataFrame(
        data={
            "n": [1, 2, 3],
            "ccy": ["USD", "JPY", "EUR"],
        },
        schema_overrides={"ccy": pl.Categorical()},
    )
    df.write_parquet(path)
    expect = df.head(1).with_columns(pl.col(pl.Categorical).cast(pl.String))

    lf = pl.scan_parquet(path)

    for predicate in [pl.col("ccy") == "USD", pl.col("ccy").is_in(["USD"])]:
        assert_frame_equal(
            lf.filter(predicate)
            .with_columns(pl.col(pl.Categorical).cast(pl.String))
            .collect(),
            expect,
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_slice_pushdown_non_zero_offset(
    tmp_path: Path, streaming: bool
) -> None:
    paths = [tmp_path / "1", tmp_path / "2", tmp_path / "3"]
    dfs = [pl.DataFrame({"x": i}) for i in range(len(paths))]

    for df, p in zip(dfs, paths, strict=True):
        df.write_parquet(p)

    # Parquet files containing only the metadata - i.e. the data parts are removed.
    # Used to test that a reader doesn't try to read any data.
    def trim_to_metadata(path: str | Path) -> None:
        path = Path(path)
        v = path.read_bytes()
        metadata_and_footer_len = 8 + int.from_bytes(v[-8:][:4], "little")
        path.write_bytes(v[-metadata_and_footer_len:])

    trim_to_metadata(paths[0])
    trim_to_metadata(paths[2])

    # Check baseline:
    # * Metadata can be read without error
    assert pl.read_parquet_schema(paths[0]) == dfs[0].schema
    # * Attempting to read any data will error
    with pytest.raises(ComputeError):
        pl.scan_parquet(paths[0]).collect(
            engine="streaming" if streaming else "in-memory"
        )

    df = dfs[1]
    assert_frame_equal(
        pl.scan_parquet(paths)
        .slice(1, 1)
        .collect(engine="streaming" if streaming else "in-memory"),
        df,
    )
    assert_frame_equal(
        pl.scan_parquet(paths[1:])
        .head(1)
        .collect(engine="streaming" if streaming else "in-memory"),
        df,
    )
    assert_frame_equal(
        (
            pl.scan_parquet([paths[1], paths[1], paths[1]])
            .with_row_index()
            .slice(1, 1)
            .collect(engine="streaming" if streaming else "in-memory")
        ),
        df.with_row_index(offset=1),
    )
    assert_frame_equal(
        (
            pl.scan_parquet([paths[1], paths[1], paths[1]])
            .with_row_index(offset=1)
            .slice(1, 1)
            .collect(engine="streaming" if streaming else "in-memory")
        ),
        df.with_row_index(offset=2),
    )
    assert_frame_equal(
        pl.scan_parquet(paths[1:])
        .head(1)
        .collect(engine="streaming" if streaming else "in-memory"),
        df,
    )

    # Negative slice unsupported in streaming
    if not streaming:
        assert_frame_equal(pl.scan_parquet(paths).slice(-2, 1).collect(), df)
        assert_frame_equal(pl.scan_parquet(paths[:2]).tail(1).collect(), df)
        assert_frame_equal(
            pl.scan_parquet(paths[1:]).slice(-99, 1).collect(), df.clear()
        )

        path = tmp_path / "data"
        df = pl.select(x=pl.int_range(0, 50))
        df.write_parquet(path)
        assert_frame_equal(pl.scan_parquet(path).slice(-100, 75).collect(), df.head(25))
        assert_frame_equal(
            pl.scan_parquet([path, path]).with_row_index().slice(-25, 100).collect(),
            pl.concat([df, df]).with_row_index().slice(75),
        )
        assert_frame_equal(
            pl.scan_parquet([path, path])
            .with_row_index(offset=10)
            .slice(-25, 100)
            .collect(),
            pl.concat([df, df]).with_row_index(offset=10).slice(75),
        )
        assert_frame_equal(
            pl.scan_parquet(path).slice(-1, (1 << 32) - 1).collect(), df.tail(1)
        )


@pytest.mark.write_disk
def test_predicate_slice_pushdown_row_index_20485(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "slice_pushdown.parquet"
    row_group_size = 100000
    num_row_groups = 3

    df = pl.select(ref=pl.int_range(num_row_groups * row_group_size))
    df.write_parquet(file_path, row_group_size=row_group_size)

    # Use a slice that starts near the end of one row group and extends into the next
    # to test handling of slices that span multiple row groups.
    slice_start = 199995
    slice_len = 10
    ldf = pl.scan_parquet(file_path)
    sliced_df = ldf.with_row_index().slice(slice_start, slice_len).collect()
    sliced_df_no_pushdown = (
        ldf.with_row_index()
        .slice(slice_start, slice_len)
        .collect(optimizations=pl.QueryOptFlags(slice_pushdown=False))
    )

    expected_index = list(range(slice_start, slice_start + slice_len))
    actual_index = list(sliced_df["index"])
    assert actual_index == expected_index

    assert_frame_equal(sliced_df, sliced_df_no_pushdown)


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_row_groups_shift_bug_18739(tmp_path: Path, streaming: bool) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.bin"

    df = pl.DataFrame({"id": range(100)})
    df.write_parquet(path, row_group_size=1)

    lf = pl.scan_parquet(path)
    assert_frame_equal(df, lf.collect(engine="streaming" if streaming else "in-memory"))


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_dsl2ir_cached_metadata(tmp_path: Path, streaming: bool) -> None:
    df = pl.DataFrame({"x": 1})
    path = tmp_path / "1"
    df.write_parquet(path)

    lf = pl.scan_parquet(path)
    assert_frame_equal(lf.collect(), df)

    # Removes the metadata portion of the parquet file.
    # Used to test that a reader doesn't try to read the metadata.
    def remove_metadata(path: str | Path) -> None:
        path = Path(path)
        v = path.read_bytes()
        metadata_and_footer_len = 8 + int.from_bytes(v[-8:][:4], "little")
        path.write_bytes(v[:-metadata_and_footer_len] + b"PAR1")

    remove_metadata(path)
    assert_frame_equal(lf.collect(engine="streaming" if streaming else "in-memory"), df)


@pytest.mark.write_disk
def test_parquet_unaligned_schema_read(tmp_path: Path) -> None:
    dfs = [
        pl.DataFrame({"a": 1, "b": 10}),
        pl.DataFrame({"b": 11, "a": 2}),
        pl.DataFrame({"x": 3, "a": 3, "y": 3, "b": 12}),
    ]

    paths = [tmp_path / "1", tmp_path / "2", tmp_path / "3"]

    for df, path in zip(dfs, paths, strict=True):
        df.write_parquet(path)

    lf = pl.scan_parquet(paths, extra_columns="ignore")

    assert_frame_equal(
        lf.select("a").collect(engine="in-memory"),
        pl.DataFrame({"a": [1, 2, 3]}),
    )

    assert_frame_equal(
        lf.with_row_index().select("a").collect(engine="in-memory"),
        pl.DataFrame({"a": [1, 2, 3]}),
    )

    assert_frame_equal(
        lf.select("b", "a").collect(engine="in-memory"),
        pl.DataFrame({"b": [10, 11, 12], "a": [1, 2, 3]}),
    )

    assert_frame_equal(
        pl.scan_parquet(paths[:2]).collect(engine="in-memory"),
        pl.DataFrame({"a": [1, 2], "b": [10, 11]}),
    )

    lf = pl.scan_parquet(paths, extra_columns="raise")

    with pytest.raises(pl.exceptions.SchemaError):
        lf.collect(engine="in-memory")

    with pytest.raises(pl.exceptions.SchemaError):
        lf.with_row_index().collect(engine="in-memory")


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_unaligned_schema_read_dtype_mismatch(
    tmp_path: Path, streaming: bool
) -> None:
    dfs = [
        pl.DataFrame({"a": 1, "b": 10}),
        pl.DataFrame({"b": "11", "a": "2"}),
    ]

    paths = [tmp_path / "1", tmp_path / "2"]

    for df, path in zip(dfs, paths, strict=True):
        df.write_parquet(path)

    lf = pl.scan_parquet(paths)

    with pytest.raises(pl.exceptions.SchemaError, match="data type mismatch"):
        lf.collect(engine="streaming" if streaming else "in-memory")


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_unaligned_schema_read_missing_cols_from_first(
    tmp_path: Path, streaming: bool
) -> None:
    dfs = [
        pl.DataFrame({"a": 1, "b": 10}),
        pl.DataFrame({"b": 11}),
    ]

    paths = [tmp_path / "1", tmp_path / "2"]

    for df, path in zip(dfs, paths, strict=True):
        df.write_parquet(path)

    lf = pl.scan_parquet(paths)

    with pytest.raises(
        (pl.exceptions.SchemaError, pl.exceptions.ColumnNotFoundError),
    ):
        lf.collect(engine="streaming" if streaming else "in-memory")


@pytest.mark.parametrize("parallel", ["columns", "row_groups", "prefiltered", "none"])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.write_disk
def test_parquet_schema_arg(
    tmp_path: Path,
    parallel: ParallelStrategy,
    streaming: bool,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    dfs = [pl.DataFrame({"a": 1, "b": 1}), pl.DataFrame({"a": 2, "b": 2})]
    paths = [tmp_path / "1", tmp_path / "2"]

    for df, path in zip(dfs, paths, strict=True):
        df.write_parquet(path)

    schema: dict[str, pl.DataType] = {
        "1": pl.Datetime(time_unit="ms", time_zone="CET"),
        "a": pl.Int64(),
        "b": pl.Int64(),
    }

    # Test `schema` containing an extra column.

    lf = pl.scan_parquet(paths, parallel=parallel, schema=schema)

    with pytest.raises((pl.exceptions.SchemaError, pl.exceptions.ColumnNotFoundError)):
        lf.collect(engine="streaming" if streaming else "in-memory")

    lf = pl.scan_parquet(
        paths, parallel=parallel, schema=schema, missing_columns="insert"
    )

    assert_frame_equal(
        lf.collect(engine="streaming" if streaming else "in-memory"),
        pl.DataFrame({"1": None, "a": [1, 2], "b": [1, 2]}, schema=schema),
    )

    # Just one test that `read_parquet` is propagating this argument.
    assert_frame_equal(
        pl.read_parquet(
            paths, parallel=parallel, schema=schema, missing_columns="insert"
        ),
        pl.DataFrame({"1": None, "a": [1, 2], "b": [1, 2]}, schema=schema),
    )

    # Issue #19081: If a schema arg is passed, ensure its fields are propagated
    # to the IR, otherwise even if `missing_columns='insert'`, downstream
    # `select()`s etc. will fail with ColumnNotFound if the column is not in
    # the first file.
    lf = pl.scan_parquet(
        paths, parallel=parallel, schema=schema, missing_columns="insert"
    ).select("1")

    s = lf.collect(engine="streaming" if streaming else "in-memory").to_series()
    assert s.len() == 2
    assert s.null_count() == 2

    # Test files containing extra columns not in `schema`

    schema: dict[str, type[pl.DataType]] = {"a": pl.Int64}  # type: ignore[no-redef]

    for missing_columns in ["insert", "raise"]:
        lf = pl.scan_parquet(
            paths,
            parallel=parallel,
            schema=schema,
            missing_columns=missing_columns,  # type: ignore[arg-type]
        )

        with pytest.raises(pl.exceptions.SchemaError):
            lf.collect(engine="streaming" if streaming else "in-memory")

    lf = pl.scan_parquet(
        paths,
        parallel=parallel,
        schema=schema,
        extra_columns="ignore",
    ).select("a")

    assert_frame_equal(
        lf.collect(engine="in-memory"),
        pl.DataFrame({"a": [1, 2]}, schema=schema),
    )

    schema: dict[str, type[pl.DataType]] = {"a": pl.Int64, "b": pl.Int8}  # type: ignore[no-redef]

    lf = pl.scan_parquet(paths, parallel=parallel, schema=schema)

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="data type mismatch for column b: incoming: Int64 != target: Int8",
    ):
        lf.collect(engine="streaming" if streaming else "in-memory")


def test_scan_parquet_empty_path_expansion(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    with pytest.raises(
        ComputeError,
        match=r"failed to retrieve first file schema \(parquet\): "
        r"expanded paths were empty \(path expansion input: "
        ".*Hint: passing a schema can allow this scan to succeed with an empty DataFrame",
    ):
        pl.scan_parquet(tmp_path).collect()

    # Scan succeeds when schema is provided
    assert_frame_equal(
        pl.scan_parquet(tmp_path, schema={"x": pl.Int64}).collect(),
        pl.DataFrame(schema={"x": pl.Int64}),
    )

    assert_frame_equal(
        pl.scan_parquet(tmp_path, schema={"x": pl.Int64}).with_row_index().collect(),
        pl.DataFrame(schema={"x": pl.Int64}).with_row_index(),
    )

    assert_frame_equal(
        pl.scan_parquet(
            tmp_path, schema={"x": pl.Int64}, hive_schema={"h": pl.String}
        ).collect(),
        pl.DataFrame(schema={"x": pl.Int64, "h": pl.String}),
    )

    assert_frame_equal(
        (
            pl.scan_parquet(
                tmp_path, schema={"x": pl.Int64}, hive_schema={"h": pl.String}
            )
            .with_row_index()
            .collect()
        ),
        pl.DataFrame(schema={"x": pl.Int64, "h": pl.String}).with_row_index(),
    )


@pytest.mark.parametrize("missing_columns", ["insert", "raise"])
@pytest.mark.write_disk
def test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249(
    tmp_path: Path,
    missing_columns: str,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    paths = [tmp_path / "1", tmp_path / "2"]

    pl.DataFrame({"a": 1, "b": 1}, schema={"a": pl.Int32, "b": pl.UInt8}).write_parquet(
        paths[0]
    )
    pl.DataFrame(
        {"a": 1, "b": 1}, schema={"a": pl.Int32, "b": pl.UInt64}
    ).write_parquet(paths[1])

    assert_frame_equal(
        pl.scan_parquet(paths, missing_columns=missing_columns)  # type: ignore[arg-type]
        .select("a")
        .collect(engine="in-memory"),
        pl.DataFrame({"a": [1, 1]}, schema={"a": pl.Int32}),
    )


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.write_disk
def test_scan_parquet_streaming_row_index_19606(
    tmp_path: Path, streaming: bool
) -> None:
    tmp_path.mkdir(exist_ok=True)
    paths = [tmp_path / "1", tmp_path / "2"]

    dfs = [pl.DataFrame({"x": i}) for i in range(len(paths))]

    for df, p in zip(dfs, paths, strict=True):
        df.write_parquet(p)

    assert_frame_equal(
        pl.scan_parquet(tmp_path)
        .with_row_index()
        .collect(engine="streaming" if streaming else "in-memory"),
        pl.DataFrame(
            {"index": [0, 1], "x": [0, 1]}, schema={"index": pl.UInt32, "x": pl.Int64}
        ),
    )


def test_scan_parquet_prefilter_panic_22452() -> None:
    # This is, the easiest way to control the threadpool size so that it is stable.
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import os

os.environ["POLARS_MAX_THREADS"] = "2"

import io

import polars as pl
from polars.testing import assert_frame_equal

assert pl.thread_pool_size() == 2

f = io.BytesIO()

df = pl.DataFrame({x: 1 for x in ["a", "b", "c", "d", "e"]})
df.write_parquet(f)
f.seek(0)

assert_frame_equal(
    pl.scan_parquet(f, parallel="prefiltered")
    .filter(pl.col(c) == 1 for c in ["a", "b", "c"])
    .collect(),
    df,
)

print("OK", end="")
""",
        ],
    )

    assert out == b"OK"


@pytest.mark.slow
def test_scan_parquet_in_mem_to_streaming_dispatch_deadlock_22641() -> None:
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import os

os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["POLARS_VERBOSE"] = "1"

import io
import sys
from threading import Thread

import polars as pl

assert pl.thread_pool_size() == 1

f = io.BytesIO()
pl.DataFrame({"x": 1}).write_parquet(f)

q = (
    pl.scan_parquet(f)
    .filter(pl.sum_horizontal(pl.col("x"), pl.col("x"), pl.col("x")) >= 0)
    .join(pl.scan_parquet(f), on="x", how="left")
)

results = [
    pl.DataFrame(),
    pl.DataFrame(),
    pl.DataFrame(),
    pl.DataFrame(),
    pl.DataFrame(),
]


def run():
    # Also test just a single scan
    pl.scan_parquet(f).collect()

    print("QUERY-FENCE", file=sys.stderr)

    results[0] = q.collect()

    print("QUERY-FENCE", file=sys.stderr)

    results[1] = pl.concat([q, q, q]).collect().head(1)

    print("QUERY-FENCE", file=sys.stderr)

    results[2] = pl.collect_all([q, q, q])[0]

    print("QUERY-FENCE", file=sys.stderr)

    results[3] = pl.collect_all(3 * [pl.concat(3 * [q])])[0].head(1)

    print("QUERY-FENCE", file=sys.stderr)

    results[4] = q.collect(background=True).fetch_blocking()


t = Thread(target=run, daemon=True)
t.start()
t.join(5)

assert [x.equals(pl.DataFrame({"x": 1})) for x in results] == [
    True,
    True,
    True,
    True,
    True,
]

print("OK", end="", file=sys.stderr)
""",
        ],
        stderr=subprocess.STDOUT,
    )

    assert out.endswith(b"OK")

    def ensure_caches_dropped(verbose_log: str) -> None:
        cache_hit_prefix = "CACHE HIT: cache id: "

        ids_hit = {
            x[len(cache_hit_prefix) :]
            for x in verbose_log.splitlines()
            if x.startswith(cache_hit_prefix)
        }

        cache_drop_prefix = "CACHE DROP: cache id: "

        ids_dropped = {
            x[len(cache_drop_prefix) :]
            for x in verbose_log.splitlines()
            if x.startswith(cache_drop_prefix)
        }

        assert ids_hit == ids_dropped

    out_str = out.decode()

    for logs in out_str.split("QUERY-FENCE"):
        ensure_caches_dropped(logs)


def test_parquet_prefiltering_inserted_column_23268() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]}, schema={"a": pl.Int8})

    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    assert_frame_equal(
        (
            pl.scan_parquet(
                f,
                schema={"a": pl.Int8, "b": pl.Int16},
                missing_columns="insert",
            )
            .filter(pl.col("a") == 3)
            .filter(pl.col("b") == 3)
            .collect()
        ),
        pl.DataFrame(schema={"a": pl.Int8, "b": pl.Int16}),
    )


@pytest.mark.may_fail_cloud  # reason: inspects logs
def test_scan_parquet_prefilter_with_cast(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    f = io.BytesIO()

    df = pl.DataFrame(
        {
            "a": ["A", "B", "C", "D", "E", "F"],
            "b": pl.Series([1, 1, 1, 1, 0, 1], dtype=pl.UInt8),
        }
    )

    df.write_parquet(f, row_group_size=3)

    md = pq.read_metadata(f)

    assert [md.row_group(i).num_rows for i in range(md.num_row_groups)] == [3, 3]

    q = pl.scan_parquet(
        f,
        schema={"a": pl.String, "b": pl.Int16},
        cast_options=pl.ScanCastOptions(integer_cast="upcast"),
        include_file_paths="file_path",
    ).filter(pl.col("b") - 1 == pl.lit(-1, dtype=pl.Int16))

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        capfd.readouterr()
        out = q.collect()
        capture = capfd.readouterr().err

    assert (
        "[ParquetFileReader]: Pre-filtered decode enabled (1 live, 1 non-live)"
        in capture
    )
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 1 / 2 row groups" in capture
    )

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "a": "E",
                "b": pl.Series([0], dtype=pl.Int16),
                "file_path": "in-mem",
            }
        ),
    )


def test_prefilter_with_n_rows_23790() -> None:
    df = pl.DataFrame(
        {
            "a": ["A", "B", "C", "D", "E", "F"],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )

    f = io.BytesIO()

    df.write_parquet(f, row_group_size=2)

    f.seek(0)

    md = pq.read_metadata(f)

    assert [md.row_group(i).num_rows for i in range(md.num_row_groups)] == [2, 2, 2]

    f.seek(0)
    q = pl.scan_parquet(f, n_rows=3).filter(pl.col("b").is_in([1, 3]))

    assert_frame_equal(q.collect(), pl.DataFrame({"a": ["A", "C"], "b": [1, 3]}))

    # With row index / file_path

    df = pl.DataFrame(
        {
            "a": ["A", "B", "C", "D", "E", "F"],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )

    f = io.BytesIO()

    df.write_parquet(f, row_group_size=2)

    f.seek(0)
    md = pq.read_metadata(f)

    assert [md.row_group(i).num_rows for i in range(md.num_row_groups)] == [2, 2, 2]

    f.seek(0)
    q = pl.scan_parquet(
        f,
        n_rows=3,
        row_index_name="index",
        include_file_paths="file_path",
    ).filter(pl.col("b").is_in([1, 3]))

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "index": pl.Series([0, 2], dtype=pl.get_index_type()),
                "a": ["A", "C"],
                "b": [1, 3],
                "file_path": "in-mem",
            }
        ),
    )


def test_scan_parquet_filter_index_panic_23849(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_PARQUET_DECODE_TARGET_VALUES_PER_THREAD", "5")
    num_rows = 3
    num_cols = 5

    f = io.BytesIO()

    pl.select(
        pl.int_range(0, num_rows).alias(f"col_{i}") for i in range(num_cols)
    ).write_parquet(f)

    for parallel in ["auto", "columns", "row_groups", "prefiltered", "none"]:
        pl.scan_parquet(f, parallel=parallel).filter(  # type: ignore[arg-type]
            pl.col("col_0").ge(0) & pl.col("col_0").lt(num_rows + 1)
        ).collect()


@pytest.mark.write_disk
def test_sink_large_rows_25834(tmp_path: Path, plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES", "1")
    df = pl.select(idx=pl.repeat(1, 20_000), bytes=pl.lit(b"AAAAA"))

    df.write_parquet(tmp_path / "single.parquet")
    assert_frame_equal(pl.scan_parquet(tmp_path / "single.parquet").collect(), df)

    md = pq.read_metadata(tmp_path / "single.parquet")
    assert [md.row_group(i).num_rows for i in range(md.num_row_groups)] == [
        16384,
        3616,
    ]

    df.write_parquet(
        tmp_path / "partitioned",
        partition_by="idx",
    )
    assert_frame_equal(pl.scan_parquet(tmp_path / "partitioned").collect(), df)


def test_scan_parquet_prefilter_is_between_non_column_input_26283() -> None:
    f = io.BytesIO()

    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=datetime(2026, 1, 1),
                end=datetime(2026, 1, 1, 0, 5, 0),
                interval="1s",
                eager=True,
            ),
        },
        schema={"timestamp": pl.Datetime("us")},
        height=301,
    )

    df.write_parquet(f)
    f.seek(0)

    q = pl.scan_parquet(f).filter(
        pl.col("timestamp")
        .dt.date()
        .cast(pl.Datetime("us"))
        .is_between(datetime(2026, 1, 1), datetime(2026, 1, 1))
    )

    assert_frame_equal(q.collect(), df)


def test_sink_parquet_arrow_schema() -> None:
    df = pl.DataFrame({"x": [0, 1, None]})

    f = io.BytesIO()
    df.lazy().sink_parquet(
        f,
        arrow_schema=pa.schema(
            [
                pa.field(
                    "x",
                    pa.int64(),
                    metadata={"custom_field_md_key": "custom_field_md_value"},
                )
            ],
        ),
    )

    f.seek(0)

    assert (
        pq.read_schema(f).field("x").metadata[b"custom_field_md_key"]
        == b"custom_field_md_value"
    )

    f = io.BytesIO()

    df.lazy().sink_parquet(
        f,
        arrow_schema=pa.schema(
            [pa.field("x", pa.int64())],
            metadata={"custom_schema_md_key": "custom_schema_md_value"},
        ),
        metadata={"custom_footer_md_key": "custom_footer_md_value"},
    )

    f.seek(0)

    assert pq.read_schema(f).metadata == {
        b"custom_schema_md_key": b"custom_schema_md_value"
    }
    assert (
        pq.read_metadata(f).metadata[b"custom_footer_md_key"]
        == b"custom_footer_md_value"
    )
    assert (
        pl.read_parquet_metadata(f)["custom_footer_md_key"] == "custom_footer_md_value"
    )
    assert pa.ipc.read_schema(
        pa.BufferReader(base64.b64decode(pq.read_metadata(f).metadata[b"ARROW:schema"]))
    ).metadata == {b"custom_schema_md_key": b"custom_schema_md_value"}

    with pytest.raises(
        SchemaError,
        match=r"provided dtype \(Int32\) does not match output dtype \(Int64\)",
    ):
        df.lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [pa.field("x", pa.int32())],
            ),
        )

    with pytest.raises(
        SchemaError,
        match="nullable is false but array contained 1 NULL",
    ):
        df.lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [pa.field("x", pa.int64(), nullable=False)],
            ),
        )

    with pytest.raises(
        SchemaError,
        match=r"schema names in arrow_schema differ",
    ):
        df.lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [pa.field("z", pa.int64())],
            ),
        )

    with pytest.raises(
        SchemaError,
        match="schema names in arrow_schema differ",
    ):
        df.lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema([]),
        )

    with pytest.raises(
        SchemaError,
        match="schema names in arrow_schema differ",
    ):
        df.lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [
                    pa.field(
                        "x",
                        pa.int64(),
                    ),
                    pa.field(
                        "y",
                        pa.int64(),
                    ),
                ],
            ),
        )


def test_sink_parquet_arrow_schema_logical_types() -> None:
    from tests.unit.datatypes.test_extension import PythonTestExtension

    df = pl.DataFrame(
        {
            "categorical": pl.Series(
                ["A"], dtype=pl.Categorical(pl.Categories.random())
            ),
            "datetime": pl.Series([datetime(2026, 1, 1)], dtype=pl.Datetime("ns")),
            "extension[str]": pl.Series(["A"], dtype=PythonTestExtension(pl.String)),
        }
    )

    with pytest.raises(SchemaError, match=r"Dictionary\(UInt32, LargeUtf8, false\)"):
        df.select("categorical").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [pa.field("categorical", pa.null())],
            ),
        )

    df.select("categorical").lazy().sink_parquet(
        io.BytesIO(),
        arrow_schema=pa.schema(
            [pa.field("categorical", pa.dictionary(pa.uint32(), pa.large_string()))],
        ),
    )

    with pytest.raises(SchemaError, match=r"Timestamp\(Nanosecond, None\)"):
        df.select("datetime").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [pa.field("datetime", pa.null())],
            ),
        )

    df.select("datetime").lazy().sink_parquet(
        io.BytesIO(),
        arrow_schema=pa.schema(
            [pa.field("datetime", pa.timestamp("ns"))],
        ),
    )

    def build_pyarrow_extension_type(name: str) -> Any:
        class PythonTestExtensionPyarrow(pa.ExtensionType):  # type: ignore[misc]
            def __init__(self, data_type: pa.DataType) -> None:
                super().__init__(data_type, name)

            def __arrow_ext_serialize__(self) -> bytes:
                return b""

            @classmethod
            def __arrow_ext_deserialize__(
                cls, storage_type: Any, serialized: Any
            ) -> Any:
                return PythonTestExtensionPyarrow(storage_type[0].type)

        return PythonTestExtensionPyarrow(pa.large_string())

    with pytest.raises(
        SchemaError,
        match=r'Extension\(ExtensionType { name: "testing.python_test_extension", inner: LargeUtf8, metadata: None }\)',
    ):
        df.select("extension[str]").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [pa.field("extension[str]", build_pyarrow_extension_type("name"))],
            ),
        )

    df.select("extension[str]").lazy().sink_parquet(
        io.BytesIO(),
        arrow_schema=pa.schema(
            [
                pa.field(
                    "extension[str]",
                    build_pyarrow_extension_type("testing.python_test_extension"),
                )
            ],
        ),
    )


def test_sink_parquet_arrow_schema_nested_types() -> None:
    df = pl.DataFrame(
        {
            "list[struct{a:int64}]": pl.Series(
                [[{"a": 1}, {"a": None}]], dtype=pl.List(pl.Struct({"a": pl.Int64}))
            ),
            "array[int64, 2]": pl.Series([[0, None]], dtype=pl.Array(pl.Int64, 2)),
        }
    )

    with pytest.raises(SchemaError, match="struct dtype mismatch"):
        df.select("list[struct{a:int64}]").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [
                    pa.field(
                        "list[struct{a:int64}]",
                        pa.large_list(pa.struct([])),
                    )
                ],
            ),
        )

    with pytest.raises(SchemaError, match="struct dtype mismatch"):
        df.select("list[struct{a:int64}]").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [
                    pa.field(
                        "list[struct{a:int64}]",
                        pa.large_list(
                            pa.struct(
                                [pa.field("a", pa.int64()), pa.field("b", pa.int64())]
                            )
                        ),
                    )
                ],
            ),
        )

    with pytest.raises(
        SchemaError,
        match="nullable is false but array contained 1 NULL",
    ):
        df.select("list[struct{a:int64}]").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [
                    pa.field(
                        "list[struct{a:int64}]",
                        pa.large_list(
                            pa.struct([pa.field("a", pa.int64(), nullable=False)])
                        ),
                    )
                ],
            ),
        )

    df.select("list[struct{a:int64}]").lazy().sink_parquet(
        io.BytesIO(),
        arrow_schema=pa.schema(
            [
                pa.field(
                    "list[struct{a:int64}]",
                    pa.large_list(pa.struct([pa.field("a", pa.int64())])),
                )
            ],
        ),
    )

    with pytest.raises(SchemaError, match="fixed-size list dtype mismatch:"):
        df.select("array[int64, 2]").lazy().sink_parquet(
            io.BytesIO(),
            arrow_schema=pa.schema(
                [
                    pa.field(
                        "array[int64, 2]",
                        pa.list_(pa.int64(), 0),
                    )
                ],
            ),
        )

    df.select("array[int64, 2]").lazy().sink_parquet(
        io.BytesIO(),
        arrow_schema=pa.schema(
            [
                pa.field(
                    "array[int64, 2]",
                    pa.list_(pa.int64(), 2),
                )
            ],
        ),
    )


def test_sink_parquet_writes_strings_as_largeutf8_by_default() -> None:
    df = pl.DataFrame({"string": "A", "binary": [b"A"]})

    with pytest.raises(
        SchemaError,
        match=r"provided dtype \(Utf8View\) does not match output dtype \(LargeUtf8\)",
    ):
        df.lazy().select("string").sink_parquet(
            io.BytesIO(), arrow_schema=pa.schema([pa.field("string", pa.string_view())])
        )

    with pytest.raises(
        SchemaError,
        match=r"provided dtype \(BinaryView\) does not match output dtype \(LargeBinary\)",
    ):
        df.lazy().select("binary").sink_parquet(
            io.BytesIO(), arrow_schema=pa.schema([pa.field("binary", pa.binary_view())])
        )

    f = io.BytesIO()

    arrow_schema = pa.schema(
        [
            pa.field("string", pa.large_string()),
            pa.field("binary", pa.large_binary()),
        ]
    )

    df.lazy().sink_parquet(f, arrow_schema=arrow_schema)

    f.seek(0)

    assert pq.read_schema(f) == arrow_schema

    f.seek(0)

    assert_frame_equal(pl.scan_parquet(f).collect(), df)


def test_sink_parquet_pyarrow_filter_string_type_26435() -> None:
    df = pl.DataFrame({"string": ["A", None, "B"], "int": [0, 1, 2]})

    f = io.BytesIO()

    df.write_parquet(f)

    f.seek(0)

    assert_frame_equal(
        pl.DataFrame(pq.read_table(f, filters=[("int", "=", 0)])),
        pl.DataFrame({"string": "A", "int": 0}),
    )

    f.seek(0)

    assert_frame_equal(
        pl.DataFrame(pq.read_table(f, filters=[("string", "=", "A")])),
        pl.DataFrame({"string": "A", "int": 0}),
    )
