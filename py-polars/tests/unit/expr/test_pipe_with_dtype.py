from __future__ import annotations

import io
import json

import pytest

import polars as pl
from polars.testing import assert_frame_equal


class UnitExtension(pl.datatypes.BaseExtension):
    """A test extension type which stores its unit in the metadata."""

    def __init__(self, unit: str) -> None:
        super().__init__(
            name="testing.pipe_with_dtype_unit",
            storage=pl.Float64,
            metadata=json.dumps({"unit": unit}),
        )


pl.register_extension_type("testing.pipe_with_dtype_unit", UnitExtension)


def to_float_if_necessary(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    return expr if dtype.is_float() else expr.cast(pl.Float64)


def test_pipe_with_dtype() -> None:
    lf = pl.LazyFrame(
        {"a": [1.0, 2.0], "b": ["1.0", "2.5"], "c": [2.0, 3.0]},
        schema={"a": pl.Float64, "b": pl.String, "c": pl.Float32},
    )

    result = lf.select(pl.all().pipe_with_dtype(to_float_if_necessary))

    assert result.collect_schema() == pl.Schema(
        {"a": pl.Float64, "b": pl.Float64, "c": pl.Float32}
    )
    assert_frame_equal(
        result.collect(),
        pl.DataFrame(
            {"a": [1.0, 2.0], "b": [1.0, 2.5], "c": [2.0, 3.0]},
            schema={"a": pl.Float64, "b": pl.Float64, "c": pl.Float32},
        ),
    )


def test_pipe_with_dtype_selector_calls_per_column() -> None:
    seen: set[tuple[str, pl.DataType]] = set()

    def record(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        seen.add((expr.meta.output_name(), dtype))
        return expr

    lf = pl.LazyFrame({"a": [1], "b": ["x"]})
    lf.select(pl.all().pipe_with_dtype(record)).collect()

    assert seen == {("a", pl.Int64()), ("b", pl.String())}


def test_pipe_with_dtype_non_column_input() -> None:
    lf = pl.LazyFrame({"a": [1, 2]})

    result = lf.select(
        (pl.col("a") > 1).pipe_with_dtype(
            lambda e, dt: e.cast(pl.Int8) if dt == pl.Boolean else e
        )
    )

    assert_frame_equal(
        result.collect(), pl.DataFrame({"a": [0, 1]}, schema={"a": pl.Int8})
    )


def test_pipe_with_dtype_in_list_eval() -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [3]]})

    result = lf.select(
        pl.col("a").list.eval(pl.element().pipe_with_dtype(to_float_if_necessary))
    )

    assert result.collect_schema() == pl.Schema({"a": pl.List(pl.Float64)})
    assert_frame_equal(
        result.collect(),
        pl.DataFrame({"a": [[1.0, 2.0], [3.0]]}),
    )


def test_pipe_with_dtype_multi_output_result() -> None:
    lf = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})

    result = lf.select(pl.col("a").pipe_with_dtype(lambda e, dt: pl.col("b", "c") + e))

    assert_frame_equal(result.collect(), pl.DataFrame({"b": [3], "c": [4]}))


def test_pipe_with_dtype_nested() -> None:
    lf = pl.LazyFrame({"a": [1, 2]})

    result = lf.select(
        pl.col("a")
        .pipe_with_dtype(lambda e, dt: e.cast(pl.Float32))
        .pipe_with_dtype(lambda e, dt: e * 2 if dt == pl.Float32 else e)
    )

    assert_frame_equal(
        result.collect(),
        pl.DataFrame({"a": [2.0, 4.0]}, schema={"a": pl.Float32}),
    )


def test_pipe_with_dtype_contexts() -> None:
    df = pl.DataFrame({"g": [1, 1, 2], "a": [1, 2, 3]})
    double = pl.col("a").pipe_with_dtype(lambda e, dt: e * 2)

    assert_frame_equal(
        df.lazy().with_columns(double).collect(),
        pl.DataFrame({"g": [1, 1, 2], "a": [2, 4, 6]}),
    )
    assert_frame_equal(
        df.lazy().filter(double > 3).collect(),
        pl.DataFrame({"g": [1, 2], "a": [2, 3]}),
    )
    assert_frame_equal(
        df.lazy().group_by("g").agg(double.sum()).sort("g").collect(),
        pl.DataFrame({"g": [1, 2], "a": [6, 6]}),
    )
    assert_frame_equal(
        df.lazy().select(double.sum().over("g")).collect(),
        pl.DataFrame({"a": [6, 6, 6]}),
    )


def test_pipe_with_dtype_extension_metadata() -> None:
    def to_meters(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        assert isinstance(dtype, UnitExtension)
        metadata = dtype.ext_metadata()
        assert metadata is not None
        factor = {"m": 1.0, "km": 1000.0}[json.loads(metadata)["unit"]]
        return expr.ext.storage() * factor

    df = pl.DataFrame(
        {"x": [1.0, 2.0], "y": [1.0, 2.0]},
        schema={"x": UnitExtension("m"), "y": UnitExtension("km")},
    )

    assert_frame_equal(
        df.select(pl.all().pipe_with_dtype(to_meters)),
        pl.DataFrame({"x": [1.0, 2.0], "y": [1000.0, 2000.0]}),
    )


def test_pipe_with_dtype_serde() -> None:
    lf = pl.LazyFrame({"a": [1], "b": ["2"]}).select(
        pl.all().pipe_with_dtype(to_float_if_necessary)
    )

    roundtripped = pl.LazyFrame.deserialize(io.BytesIO(lf.serialize()))

    assert_frame_equal(roundtripped.collect(), lf.collect())


def test_pipe_with_dtype_without_schema() -> None:
    # "Is this a scalar?" can't be answered without knowing the input dtype,
    # because the answer depends on which expression the callback returns.

    expr = pl.col("a").pipe_with_dtype(lambda e, dt: e.sum())
    with pytest.raises(
        pl.exceptions.ColumnNotFoundError,
        match="'pipe_with_dtype' failed to resolve its input dtype",
    ):
        expr.meta.is_scalar()


def test_pipe_with_dtype_raises_at_plan_time() -> None:
    def fail(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        msg = "planning failed"
        raise ValueError(msg)

    lf = pl.LazyFrame({"a": [1]}).select(pl.col("a").pipe_with_dtype(fail))

    with pytest.raises(ValueError, match="planning failed"):
        lf.collect()
