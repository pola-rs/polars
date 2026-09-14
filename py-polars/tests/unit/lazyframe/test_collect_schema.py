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


@pytest.mark.parametrize(
    "lhs_dtype",
    [pl.Boolean, pl.Int64, pl.Float16, pl.Float32, pl.Float64, pl.Decimal(10, 2)],
)
@pytest.mark.parametrize(
    "rhs_dtype",
    [pl.Duration, pl.Time, pl.Date, pl.Datetime],
)
def test_collect_schema_truediv_rejects_temporal_rhs_27565(
    lhs_dtype: pl.DataType, rhs_dtype: pl.DataType
) -> None:
    lf = pl.LazyFrame(
        {"a": [None], "b": [None]}, schema={"a": lhs_dtype, "b": rhs_dtype}
    ).select(result=pl.col("a") / pl.col("b"))
    with pytest.raises(pl.exceptions.InvalidOperationError, match="not allowed"):
        lf.collect_schema()


@pytest.mark.parametrize(
    "lhs_dtype",
    [pl.Duration, pl.Time, pl.Date, pl.Datetime],
)
def test_collect_schema_truediv_rejects_temporal_div_string_27565(
    lhs_dtype: pl.DataType,
) -> None:
    lf = (
        pl.LazyFrame({"a": [0], "b": ["x"]})
        .cast({"a": lhs_dtype})
        .select(result=pl.col("a") / pl.col("b"))
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="not allowed"):
        lf.collect_schema()


def test_collect_schema_truediv_duration_numeric_ok_27565() -> None:
    lf = (
        pl.LazyFrame({"a": [100], "b": [2]})
        .cast({"a": pl.Duration})
        .select(result=pl.col("a") / pl.col("b"))
    )
    assert lf.collect_schema()["result"] == pl.Duration()


def test_collect_schema_truediv_duration_by_duration_ok_27565() -> None:
    lf = (
        pl.LazyFrame({"a": [100], "b": [50]})
        .cast({"a": pl.Duration, "b": pl.Duration})
        .select(result=pl.col("a") / pl.col("b"))
    )
    assert lf.collect_schema()["result"] == pl.Float64
