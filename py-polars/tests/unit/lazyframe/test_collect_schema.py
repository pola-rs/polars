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


# https://github.com/pola-rs/polars/issues/27565 — collect_schema() must
# raise InvalidOperationError for unsupported input dtypes instead of
# silently returning a dtype that the actual `collect()` would reject.


@pytest.mark.parametrize("op", ["abs", "neg"])
def test_collect_schema_abs_neg_reject_string_27565(op: str) -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        result=getattr(pl.col("a"), op)()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match=op):
        lf.collect_schema()


def test_collect_schema_pow_rejects_string_base_27565() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        result=pl.col("a") ** 2
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match=r"`pow`.*base"):
        lf.collect_schema()


def test_collect_schema_pow_rejects_string_exponent_27565() -> None:
    lf = pl.LazyFrame(
        {"a": [1], "b": [None]}, schema={"a": pl.Int64, "b": pl.String}
    ).select(result=pl.col("a") ** pl.col("b"))
    with pytest.raises(pl.exceptions.InvalidOperationError, match=r"`pow`.*exponent"):
        lf.collect_schema()


@pytest.mark.parametrize("fn", ["sqrt", "cbrt"])
def test_collect_schema_sqrt_cbrt_returns_float_27565(fn: str) -> None:
    # Case 4 of #27565: `sqrt(String)` used to return `String` in collect_schema()
    # while collect() returned `Float64`. The fix makes schema match runtime by
    # always returning Float64 (including for non-numeric inputs that runtime
    # silently coerces to null).
    for dtype in (pl.String, pl.Int64, pl.Float64):
        lf = pl.LazyFrame({"a": [None]}, schema={"a": dtype}).select(
            result=getattr(pl.col("a"), fn)()
        )
        assert lf.collect_schema()["result"] == pl.Float64


def test_collect_schema_trig_rejects_string_27565() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        result=pl.col("a").sin()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="trigonometry"):
        lf.collect_schema()


@pytest.mark.parametrize("dtype", [pl.String, pl.Boolean])
def test_collect_schema_entropy_rejects_non_numeric_27565(
    dtype: pl.DataType,
) -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": dtype}).select(
        result=pl.col("a").entropy()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="entropy"):
        lf.collect_schema()


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
    ("rhs_dtype", "rhs_val"),
    [
        (pl.Duration, 0),
        (pl.Time, 0),
        (pl.Date, 0),
        (pl.Datetime, 0),
    ],
)
def test_collect_schema_truediv_rejects_numeric_div_temporal_27565(
    rhs_dtype: pl.DataType, rhs_val: int
) -> None:
    lf = (
        pl.LazyFrame({"a": [1], "b": [rhs_val]})
        .cast({"b": rhs_dtype})
        .select(result=pl.col("a") / pl.col("b"))
    )
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
