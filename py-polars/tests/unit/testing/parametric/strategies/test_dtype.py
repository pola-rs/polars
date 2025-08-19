import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars.testing.parametric.strategies.core import dataframes
from polars.testing.parametric.strategies.dtype import dtypes


@given(dtype=dtypes())
def test_dtypes(dtype: pl.DataType) -> None:
    assert isinstance(dtype, pl.DataType)


@given(dtype=dtypes(nesting_level=0))
def test_dtypes_nesting_level(dtype: pl.DataType) -> None:
    assert not dtype.is_nested()


@given(dtype=dtypes(allowed_dtypes=[pl.Datetime], allow_time_zones=False))
def test_dtypes_allow_time_zones(dtype: pl.DataType) -> None:
    assert getattr(dtype, "time_zone", None) is None


@given(st.data())
def test_dtypes_allowed(data: st.DataObject) -> None:
    allowed_dtype = data.draw(dtypes())
    result = data.draw(dtypes(allowed_dtypes=[allowed_dtype]))
    assert result == allowed_dtype


@given(st.data())
def test_dtypes_excluded(data: st.DataObject) -> None:
    excluded_dtype = data.draw(dtypes())
    result = data.draw(dtypes(excluded_dtypes=[excluded_dtype]))
    assert result != excluded_dtype


@given(dtype=dtypes(allowed_dtypes=[pl.Duration], excluded_dtypes=[pl.Duration("ms")]))
def test_dtypes_allowed_excluded_instance(dtype: pl.DataType) -> None:
    assert isinstance(dtype, pl.Duration)
    assert dtype.time_unit != "ms"


@given(
    dtype=dtypes(
        allowed_dtypes=[pl.Duration("ns"), pl.Date], excluded_dtypes=[pl.Duration]
    )
)
def test_dtypes_allowed_excluded_priority(dtype: pl.DataType) -> None:
    assert dtype == pl.Date


@given(dtype=dtypes(allowed_dtypes=[pl.Int8(), pl.Duration("ms")]))
def test_dtypes_allowed_instantiated(dtype: pl.DataType) -> None:
    assert dtype in (pl.Int8(), pl.Duration("ms"))


@given(dtype=dtypes(allowed_dtypes=[pl.List(pl.List), pl.Int64]))
def test_dtypes_allowed_uninstantiated_nested(dtype: pl.DataType) -> None:
    assert dtype in (pl.List, pl.Int64)


@pytest.mark.parametrize(
    "expr",
    [
        # TODO: Add more (bitwise) operators once their types are resolved correctly
        pl.col("col0") > pl.col("col1"),
        pl.col("col0") >= pl.col("col1"),
        pl.col("col0") < pl.col("col1"),
        pl.col("col0") <= pl.col("col1"),
        pl.col("col0") == pl.col("col1"),
        pl.col("col0") != pl.col("col1"),
        pl.col("col0") + pl.col("col1"),
        pl.col("col0") - pl.col("col1"),
        pl.col("col0") * pl.col("col1"),
        pl.col("col0") / pl.col("col1"),
        pl.col("col0").truediv(pl.col("col1")),
        pl.col("col0") // pl.col("col1"),
        pl.col("col0") % pl.col("col1"),
    ],
)
@given(
    df=dataframes(
        excluded_dtypes=[
            # TODO: At the moment, these dtypes are known to be mismatched
            # between the in-memory engine and the planner.
            # We should include more types going forward.
            pl.Struct,
            pl.Decimal,
            pl.Categorical,
            pl.Enum,
            pl.Null,
            pl.Binary,
        ],
        cols=2,
        min_size=1,
        max_size=1,
    ),
)
def test_lazy_collect_schema_matches_computed_schema(
    expr: pl.Expr, df: pl.DataFrame
) -> None:
    lazy_df = df.lazy().select(expr)

    expected_schema = None
    try:
        expected_schema = lazy_df.collect().schema
    except (
        pl.exceptions.InvalidOperationError,
        pl.exceptions.SchemaError,
        pl.exceptions.ComputeError,
    ):
        return

    actual_schema = lazy_df.collect_schema()
    assert actual_schema == expected_schema, (
        f"{expr} on {df.dtypes} results in {actual_schema} instead of {expected_schema}\n"
        f"result of computation is:\n{lazy_df.collect()}\n"
    )
