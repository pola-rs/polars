from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


def test_reverse_list_22829() -> None:
    schema = {"a": pl.Int64(), "b": pl.List(pl.Binary)}
    df = pl.DataFrame([], schema=schema, schema_overrides=schema)
    assert df.schema == df.reverse().schema


def test_reverse_df() -> None:
    out = pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).reverse()
    expected = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    assert_frame_equal(out.collect(), expected)


def test_reverse_series() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5])
    assert s.reverse().to_list() == [5, 4, 3, 2, 1]

    s = pl.Series("values", ["a", "b", None, "y", "x"])
    assert s.reverse().to_list() == ["x", "y", None, "b", "a"]


def test_reverse_binary() -> None:
    # single chunk
    s = pl.Series("values", ["a", "b", "c", "d"]).cast(pl.Binary)
    assert s.reverse().to_list() == [b"d", b"c", b"b", b"a"]

    # multiple chunks
    chunk1 = pl.Series("values", ["a", "b"])
    chunk2 = pl.Series("values", ["c", "d"])
    s = chunk1.extend(chunk2).cast(pl.Binary)
    assert s.n_chunks() == 2
    assert s.reverse().to_list() == [b"d", b"c", b"b", b"a"]


@pytest.mark.parametrize(
    "dtype", [pl.List(pl.Int64), pl.List(pl.Map(pl.String, pl.Int64))]
)
def test_reverse_all_null_list_preserves_dtype(dtype: pl.DataType) -> None:
    s = pl.Series("values", [None, None], dtype=dtype)
    assert_series_equal(s.reverse(), s)

    s = pl.concat([s, s], rechunk=False)
    assert s.n_chunks() == 2
    assert_series_equal(s.reverse(), s)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_reverse_diagonal_concat_preserves_schema_28409(engine: EngineType) -> None:
    a = pl.LazyFrame({"L": [[1, 2]], "b": [1]})
    b = pl.LazyFrame({"z": [7]})
    q = pl.concat([a.sort("b").filter(pl.col("b") < -9), b], how="diagonal").reverse()
    expected = pl.DataFrame(
        {"L": [None], "b": [None], "z": [7]},
        schema={"L": pl.List(pl.Int64), "b": pl.Int64, "z": pl.Int64},
    )
    assert q.collect_schema() == expected.schema
    assert_frame_equal(q.collect(engine=engine), expected)

    q = q.filter(pl.col("b") > 5)
    assert q.collect_schema() == expected.schema
    assert_frame_equal(q.collect(engine=engine), expected.clear())


def test_reverse_list_preserves_extension_dtype() -> None:
    name = "testing.reverse_list"
    pl.register_extension_type(name, pl.Extension)
    try:
        dtype = pl.List(pl.Extension(name, pl.String, "metadata"))
        s = pl.Series("values", [["a"], None, [], ["b"]]).cast(dtype)
        expected = pl.Series("values", [["b"], [], None, ["a"]]).cast(dtype)
        assert_series_equal(s.reverse(), expected)
    finally:
        pl.unregister_extension_type(name)
