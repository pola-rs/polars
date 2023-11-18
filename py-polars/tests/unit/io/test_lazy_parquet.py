from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.type_aliases import ParallelStrategy


@pytest.fixture()
def parquet_file_path(io_files_path: Path) -> Path:
    return io_files_path / "small.parquet"


@pytest.fixture()
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


def test_row_count(foods_parquet_path: Path) -> None:
    df = pl.read_parquet(foods_parquet_path, row_count_name="row_count")
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_parquet(foods_parquet_path, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_parquet(foods_parquet_path, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


@pytest.mark.write_disk()
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


@pytest.mark.write_disk()
def test_null_parquet(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    file_path = tmp_path / "null.parquet"
    df.write_parquet(file_path)
    out = pl.read_parquet(file_path)
    assert_frame_equal(out, df)


@pytest.mark.write_disk()
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


@pytest.mark.write_disk()
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


@pytest.mark.write_disk()
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


def test_row_count_schema_parquet(parquet_file_path: Path) -> None:
    assert (
        pl.scan_parquet(str(parquet_file_path), row_count_name="id")
        .select(["id", "b"])
        .collect()
    ).dtypes == [pl.UInt32, pl.Utf8]


@pytest.mark.write_disk()
def test_parquet_eq_statistics(monkeypatch: Any, capfd: Any, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    monkeypatch.setenv("POLARS_VERBOSE", "1")

    df = pl.DataFrame({"idx": pl.arange(100, 200, eager=True)}).with_columns(
        (pl.col("idx") // 25).alias("part")
    )
    df = pl.concat(df.partition_by("part", as_dict=False), rechunk=False)
    assert df.n_chunks("all") == [4, 4]

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False)

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False)

    for streaming in [False, True]:
        for pred in [
            pl.col("idx") == 50,
            pl.col("idx") == 150,
            pl.col("idx") == 210,
        ]:
            result = (
                pl.scan_parquet(file_path).filter(pred).collect(streaming=streaming)
            )
            assert_frame_equal(result, df.filter(pred))

        captured = capfd.readouterr().err
        assert (
            "parquet file must be read, statistics not sufficient for predicate."
            in captured
        )
        assert (
            "parquet file can be skipped, the statistics were sufficient"
            " to apply the predicate." in captured
        )


@pytest.mark.write_disk()
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
        "parquet file must be read, statistics not sufficient for predicate."
        in captured
    )
    assert (
        "parquet file can be skipped, the statistics were sufficient"
        " to apply the predicate." in captured
    )


@pytest.mark.write_disk()
def test_parquet_statistics(monkeypatch: Any, capfd: Any, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    monkeypatch.setenv("POLARS_VERBOSE", "1")

    df = pl.DataFrame({"idx": pl.arange(0, 100, eager=True)}).with_columns(
        (pl.col("idx") // 25).alias("part")
    )
    df = pl.concat(df.partition_by("part", as_dict=False), rechunk=False)
    assert df.n_chunks("all") == [4, 4]

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False)

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
        "parquet file must be read, statistics not sufficient for predicate."
        in captured
    )
    assert (
        "parquet file can be skipped, the statistics were sufficient"
        " to apply the predicate." in captured
    )


@pytest.mark.write_disk()
def test_streaming_categorical(tmp_path: Path) -> None:
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


@pytest.mark.write_disk()
def test_parquet_struct_categorical(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame(
        [
            pl.Series("a", ["bob"], pl.Categorical),
            pl.Series("b", ["foo"], pl.Categorical),
        ]
    )

    file_path = tmp_path / "categorical.parquet"
    df.write_parquet(file_path)

    with pl.StringCache():
        out = pl.read_parquet(file_path).select(pl.col("b").value_counts())
    assert out.to_dict(as_series=False) == {"b": [{"b": "foo", "counts": 1}]}


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


@pytest.mark.write_disk()
def test_parquet_statistics_filter_9925(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "codes.parquet"
    df = pl.DataFrame({"code": [300964, 300972, 500_000, 26]})
    df.write_parquet(file_path, statistics=True)

    q = pl.scan_parquet(file_path).filter(
        (pl.col("code").floordiv(100_000)).is_in([0, 3])
    )
    assert q.collect().to_dict(as_series=False) == {"code": [300964, 300972, 26]}


@pytest.mark.write_disk()
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


@pytest.mark.write_disk()
def test_parquet_many_row_groups_12297(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "foo.parquet"
    df = pl.DataFrame({"x": range(100)})
    df.write_parquet(file_path, row_group_size=5, use_pyarrow=True)
    assert_frame_equal(pl.scan_parquet(file_path).collect(), df)


@pytest.mark.write_disk()
def test_row_count_empty_file(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "test.parquet"
    pl.DataFrame({"a": []}).write_parquet(file_path)
    assert pl.scan_parquet(file_path).with_row_count(
        "idx"
    ).collect().schema == OrderedDict([("idx", pl.UInt32), ("a", pl.Float32)])


@pytest.mark.write_disk()
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


@pytest.mark.write_disk()
def test_parquet_different_schema(tmp_path: Path) -> None:
    # Schema is different but the projected columns are same dtype.
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    a = pl.DataFrame({"a": [1.0], "b": "a"})

    b = pl.DataFrame({"a": [1], "b": "a"})

    a.write_parquet(f1)
    b.write_parquet(f2)
    assert pl.scan_parquet([f1, f2]).select("b").collect().columns == ["b"]
