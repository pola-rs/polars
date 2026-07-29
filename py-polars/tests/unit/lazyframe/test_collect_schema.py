from datetime import date, time, timedelta
from decimal import Decimal

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
    "expr",
    [
        pl.col("a").rolling_mean(2),
        pl.col("a").rolling_median(2),
        pl.col("a").rolling_std(2),
        pl.col("a").rolling_var(2),
        pl.col("a").ewm_mean(alpha=0.5),
        pl.col("a").ewm_std(alpha=0.5),
        pl.col("a").ewm_var(alpha=0.5),
    ],
)
def test_collect_schema_rolling_ewm_string_28564(expr: pl.Expr) -> None:
    q = pl.LazyFrame({"a": ["1", "2", "3"]}).select(expr)

    assert q.collect_schema() == q.collect().schema == {"a": pl.Float64}


@pytest.mark.parametrize(
    ("series", "expr"),
    [
        (
            pl.Series("a", [date(2020, 1, 1)]),
            pl.col("a").ewm_mean(alpha=0.5),
        ),
        (
            pl.Series("a", [timedelta(seconds=1)], dtype=pl.Duration("us")),
            pl.col("a").rolling_std(1),
        ),
        (pl.Series("a", [time(12)]), pl.col("a").ewm_var(alpha=0.5)),
    ],
)
def test_collect_schema_rolling_ewm_logical_28564(
    series: pl.Series, expr: pl.Expr
) -> None:
    q = pl.LazyFrame(series).select(expr)

    assert q.collect_schema() == q.collect().schema == {"a": pl.Float64}


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").rolling_std(2),
        pl.col("a").rolling_var(2),
        pl.col("a").ewm_mean(alpha=0.5),
    ],
)
def test_collect_schema_rolling_ewm_float32_28564(expr: pl.Expr) -> None:
    q = pl.LazyFrame({"a": pl.Series([1.0, 2.0], dtype=pl.Float32)}).select(expr)

    assert q.collect_schema() == q.collect().schema == {"a": pl.Float32}


def test_collect_schema_rolling_temporal_preserved_28564() -> None:
    q = pl.LazyFrame({"a": [date(2020, 1, 1)]}).select(pl.col("a").rolling_mean(1))

    assert q.collect_schema() == q.collect().schema == {"a": pl.Datetime("us")}


@pytest.mark.parametrize(
    "expr", [pl.col("a").rolling_var(1), pl.col("a").ewm_var(alpha=0.5)]
)
def test_collect_schema_duration_var_unsupported_28564(expr: pl.Expr) -> None:
    q = pl.LazyFrame(
        {"a": pl.Series([timedelta(seconds=1)], dtype=pl.Duration("us"))}
    ).select(expr)

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect_schema()


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", ["1", "2"], dtype=pl.Categorical),
        pl.Series("a", [b"1", b"2"], dtype=pl.Binary),
    ],
)
def test_collect_schema_rolling_unsupported_28564(series: pl.Series) -> None:
    q = pl.LazyFrame(series).select(pl.col("a").rolling_mean(2))

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect_schema()


@pytest.mark.parametrize("expr", [pl.col("a") / 2, pl.col("a").cum_sum()])
def test_collect_schema_decimal_precision_28564(expr: pl.Expr) -> None:
    series = pl.Series("a", [Decimal("1.5")] * 3, dtype=pl.Decimal(10, 2))
    q = pl.LazyFrame(series).select(expr)

    assert q.collect_schema() == q.collect().schema == {"a": pl.Decimal(38, 2)}


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Int64, pl.Decimal(38, 2)),
        (pl.Float16, pl.Float64),
        (pl.Float32, pl.Float64),
    ],
)
def test_collect_schema_decimal_rhs_28564(
    dtype: pl.DataType, expected: pl.DataType
) -> None:
    q = pl.LazyFrame(
        {
            "a": pl.Series([2], dtype=dtype),
            "b": pl.Series([Decimal("1.5")], dtype=pl.Decimal(10, 2)),
        }
    ).select(pl.col("a") / pl.col("b"))

    assert q.collect_schema() == q.collect().schema == {"a": expected}


def test_collect_schema_decimal_cum_prod_28564() -> None:
    series = pl.Series("a", [Decimal("1.5")], dtype=pl.Decimal(10, 2))
    q = pl.LazyFrame(series).select(pl.col("a").cum_prod())

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect_schema()
