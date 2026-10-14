import pytest
from hypothesis import given

import polars as pl
from polars.testing.parametric import dataframes


@given(lf=dataframes(lazy=True))
def test_collect_schema_parametric(lf: pl.LazyFrame) -> None:
    assert lf.collect_schema() == lf.collect().schema


def test_collect_schema() -> None:
    lf = pl.LazyFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    result = lf.collect_schema()
    expected = pl.Schema({"foo": pl.Int64(), "bar": pl.Float64(), "ham": pl.String()})
    assert result == expected


def test_collect_schema_with_row_index_duplicate() -> None:
    lf = pl.LazyFrame({"index": []}).with_row_index()
    with pytest.raises(
        pl.exceptions.DuplicateError, match="duplicate column name index"
    ):
        _ = lf.collect_schema()

    lf = pl.LazyFrame({}).with_row_index().with_row_index()
    with pytest.raises(
        pl.exceptions.DuplicateError, match="duplicate column name index"
    ):
        _ = lf.collect_schema()


def test_collect_schema_unpivot_duplicate() -> None:
    lf = pl.LazyFrame({"variable": [], "a": []}).unpivot(["a"])
    with pytest.raises(
        pl.exceptions.DuplicateError, match="duplicate column name 'variable'"
    ):
        _ = lf.collect_schema()

    lf = pl.LazyFrame({"value": [], "a": []}).unpivot(["a"])
    with pytest.raises(
        pl.exceptions.DuplicateError, match="duplicate column name 'value'"
    ):
        _ = lf.collect_schema()


def test_arr_get_oob_errors_at_schema_26088() -> None:
    lf = pl.LazyFrame({"arr": [[1, 2, 3]]}).cast({"arr": pl.Array(pl.Int64, shape=3)})

    with pytest.raises(pl.exceptions.ComputeError):
        lf.select(pl.col("arr").arr.get(5)).collect_schema()

    with pytest.raises(pl.exceptions.ComputeError):
        lf.select(pl.col("arr").arr.get(-4)).collect_schema()

    lf.select(pl.col("arr").arr.get(2)).collect_schema()

    lf.select(pl.col("arr").arr.get(-1)).collect_schema()

    lf.select(pl.col("arr").arr.get(5, null_on_oob=True)).collect_schema()


def test_collect_schema_deep_chain() -> None:
    # Each step reuses the resolved plan of its input, this used to be cubic in the
    # length of the chain.
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    for i in range(1000):
        lf = lf.with_columns((pl.col("a") + i).alias("a"))
        assert lf.collect_schema() == {"a": pl.Int64}

    lf = lf.select(pl.col("a").sum())
    assert lf.collect_schema() == {"a": pl.Int64}


def test_collect_schema_derived_frames_are_independent() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}).with_columns(b=pl.lit(1))
    assert lf.collect_schema() == {"a": pl.Int64, "b": pl.Int32}

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        lf.select("missing").collect_schema()
    assert lf.select().collect_schema() == {}

    assert lf.collect_schema() == {"a": pl.Int64, "b": pl.Int32}
    child = lf.with_columns(c=pl.lit("x"))
    assert child.collect_schema() == {"a": pl.Int64, "b": pl.Int32, "c": pl.String}
    assert child.collect().shape == (3, 3)
    assert lf.collect().shape == (3, 2)


def test_collect_schema_reentrant_from_pipe_with_schema() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    lf.collect_schema()
    child = lf.with_columns(b=pl.col("a") * 2)
    child.collect_schema()

    def callback(inner: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        # Collect schemas of frames sharing cached state with the one being resolved.
        assert lf.with_columns(c=pl.lit(1)).collect_schema().names() == ["a", "c"]
        assert lf.collect_schema().names() == ["a"]
        return inner.with_columns(pl.lit(len(schema)).alias("n"))

    piped = child.pipe_with_schema(callback)
    assert piped.collect_schema().names() == ["a", "b", "n"]
    assert piped.collect()["n"].to_list() == [2, 2, 2]
    assert child.collect_schema().names() == ["a", "b"]
