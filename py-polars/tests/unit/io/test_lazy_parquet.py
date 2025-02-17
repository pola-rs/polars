from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import ParallelStrategy


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
    monkeypatch: Any, foods_parquet_path: Path
) -> None:
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
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
@pytest.mark.usefixtures("test_global_and_local")
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
    ).dtypes == [pl.UInt32, pl.String]


@pytest.mark.write_disk
def test_parquet_is_in_statistics(monkeypatch: Any, capfd: Any, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    monkeypatch.setenv("POLARS_VERBOSE", "1")

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
    assert (
        "parquet row group must be read, statistics not sufficient for predicate."
        in captured
    )
    assert (
        "parquet row group can be skipped, the statistics were sufficient to apply the predicate."
        in captured
    )


@pytest.mark.write_disk
def test_parquet_statistics(monkeypatch: Any, capfd: Any, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    monkeypatch.setenv("POLARS_VERBOSE", "1")

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
    assert (
        "parquet row group must be read, statistics not sufficient for predicate."
        in captured
    )
    assert (
        "parquet row group can be skipped, the statistics were sufficient"
        " to apply the predicate." in captured
    )


@pytest.mark.write_disk
@pytest.mark.usefixtures("test_global_and_local")
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

    with pl.StringCache():
        result = (
            pl.scan_parquet(file_path)
            .group_by("name")
            .agg(pl.col("amount").sum())
            .collect()
            .sort("name")
        )
        expected = pl.DataFrame(
            {"name": ["Bob", "Alice"], "amount": [400, 200]},
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
        streaming=streaming
    ).columns == ["b"]


@pytest.mark.write_disk
def test_nested_slice_12480(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    df = pl.select(pl.lit(1).repeat_by(10_000).explode().cast(pl.List(pl.Int32)))

    df.write_parquet(path, use_pyarrow=True, pyarrow_options={"data_page_size": 1})

    assert pl.scan_parquet(path).slice(0, 1).collect().height == 1


@pytest.mark.write_disk
def test_scan_deadlock_rayon_spawn_from_async_15172(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    monkeypatch.setenv("POLARS_MAX_THREADS", "1")
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

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        pl.scan_parquet(tmp_path).collect(streaming=streaming)


@pytest.mark.write_disk
def test_predicate_push_down_categorical_17744(tmp_path: Path) -> None:
    path = tmp_path / "1"

    df = pl.DataFrame(
        data={
            "n": [1, 2, 3],
            "ccy": ["USD", "JPY", "EUR"],
        },
        schema_overrides={"ccy": pl.Categorical("lexical")},
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

    for df, p in zip(dfs, paths):
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
        pl.scan_parquet(paths[0]).collect(streaming=streaming)

    df = dfs[1]
    assert_frame_equal(
        pl.scan_parquet(paths).slice(1, 1).collect(streaming=streaming), df
    )
    assert_frame_equal(
        pl.scan_parquet(paths[1:]).head(1).collect(streaming=streaming), df
    )
    assert_frame_equal(
        (
            pl.scan_parquet([paths[1], paths[1], paths[1]])
            .with_row_index()
            .slice(1, 1)
            .collect(streaming=streaming)
        ),
        df.with_row_index(offset=1),
    )
    assert_frame_equal(
        (
            pl.scan_parquet([paths[1], paths[1], paths[1]])
            .with_row_index(offset=1)
            .slice(1, 1)
            .collect(streaming=streaming)
        ),
        df.with_row_index(offset=2),
    )
    assert_frame_equal(
        pl.scan_parquet(paths[1:]).head(1).collect(streaming=streaming), df
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
        ldf.with_row_index().slice(slice_start, slice_len).collect(slice_pushdown=False)
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
    assert_frame_equal(df, lf.collect(streaming=streaming))


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
    assert_frame_equal(lf.collect(streaming=streaming), df)


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [True, False])
def test_parquet_unaligned_schema_read(tmp_path: Path, streaming: bool) -> None:
    dfs = [
        pl.DataFrame({"a": 1, "b": 10}),
        pl.DataFrame({"b": 11, "a": 2}),
        pl.DataFrame({"x": 3, "a": 3, "y": 3, "b": 12}),
    ]

    paths = [tmp_path / "1", tmp_path / "2", tmp_path / "3"]

    for df, path in zip(dfs, paths):
        df.write_parquet(path)

    lf = pl.scan_parquet(paths)

    assert_frame_equal(
        lf.select("a").collect(streaming=streaming), pl.DataFrame({"a": [1, 2, 3]})
    )

    assert_frame_equal(
        lf.select("b", "a").collect(streaming=streaming),
        pl.DataFrame({"b": [10, 11, 12], "a": [1, 2, 3]}),
    )

    assert_frame_equal(
        pl.scan_parquet(paths[:2]).collect(streaming=streaming),
        pl.DataFrame({"a": [1, 2], "b": [10, 11]}),
    )

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="parquet file contained extra columns and no selection was given",
    ):
        lf.collect(streaming=streaming)


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

    for df, path in zip(dfs, paths):
        df.write_parquet(path)

    lf = pl.scan_parquet(paths)

    with pytest.raises(pl.exceptions.SchemaError, match="data type mismatch"):
        lf.collect(streaming=streaming)


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

    for df, path in zip(dfs, paths):
        df.write_parquet(path)

    lf = pl.scan_parquet(paths)

    with pytest.raises(
        pl.exceptions.ColumnNotFoundError,
        match="did not find column in file: a",
    ):
        lf.collect(streaming=streaming)


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

    for df, path in zip(dfs, paths):
        df.write_parquet(path)

    schema: dict[str, pl.DataType] = {
        "1": pl.Datetime(time_unit="ms", time_zone="CET"),
        "a": pl.Int64(),
        "b": pl.Int64(),
    }

    # Test `schema` containing an extra column.

    lf = pl.scan_parquet(paths, parallel=parallel, schema=schema)

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        lf.collect(streaming=streaming)

    lf = pl.scan_parquet(
        paths, parallel=parallel, schema=schema, allow_missing_columns=True
    )

    assert_frame_equal(
        lf.collect(streaming=streaming),
        pl.DataFrame({"1": None, "a": [1, 2], "b": [1, 2]}, schema=schema),
    )

    # Just one test that `read_parquet` is propagating this argument.
    assert_frame_equal(
        pl.read_parquet(
            paths, parallel=parallel, schema=schema, allow_missing_columns=True
        ),
        pl.DataFrame({"1": None, "a": [1, 2], "b": [1, 2]}, schema=schema),
    )

    # Issue #19081: If a schema arg is passed, ensure its fields are propagated
    # to the IR, otherwise even if `allow_missing_columns=True`, downstream
    # `select()`s etc. will fail with ColumnNotFound if the column is not in
    # the first file.
    lf = pl.scan_parquet(
        paths, parallel=parallel, schema=schema, allow_missing_columns=True
    ).select("1")

    s = lf.collect(streaming=streaming).to_series()
    assert s.len() == 2
    assert s.null_count() == 2

    # Test files containing extra columns not in `schema`

    schema: dict[str, type[pl.DataType]] = {"a": pl.Int64}  # type: ignore[no-redef]

    for allow_missing_columns in [True, False]:
        lf = pl.scan_parquet(
            paths,
            parallel=parallel,
            schema=schema,
            allow_missing_columns=allow_missing_columns,
        )

        with pytest.raises(
            pl.exceptions.SchemaError, match="file contained extra columns"
        ):
            lf.collect(streaming=streaming)

    lf = pl.scan_parquet(paths, parallel=parallel, schema=schema).select("a")

    assert_frame_equal(
        lf.collect(streaming=streaming),
        pl.DataFrame({"a": [1, 2]}, schema=schema),
    )

    schema: dict[str, type[pl.DataType]] = {"a": pl.Int64, "b": pl.Int8}  # type: ignore[no-redef]

    lf = pl.scan_parquet(paths, parallel=parallel, schema=schema)

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="data type mismatch for column b: expected: i8, found: i64",
    ):
        lf.collect(streaming=streaming)


def test_scan_parquet_schema_specified_with_empty_files_list(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

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


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("allow_missing_columns", [True, False])
@pytest.mark.write_disk
def test_scan_parquet_ignores_dtype_mismatch_for_non_projected_columns_19249(
    tmp_path: Path,
    allow_missing_columns: bool,
    streaming: bool,
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
        pl.scan_parquet(paths, allow_missing_columns=allow_missing_columns)
        .select("a")
        .collect(streaming=streaming),
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

    for df, p in zip(dfs, paths):
        df.write_parquet(p)

    assert_frame_equal(
        pl.scan_parquet(tmp_path).with_row_index().collect(streaming=streaming),
        pl.DataFrame(
            {"index": [0, 1], "x": [0, 1]}, schema={"index": pl.UInt32, "x": pl.Int64}
        ),
    )
