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


@pytest.mark.parametrize("op", ["abs", "neg"])
def test_collect_schema_abs_neg_reject_string_27565(op: str) -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        result=getattr(pl.col("a"), op)()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match=op):
        lf.collect_schema()


@pytest.mark.parametrize("dtype", [pl.UInt8, pl.Int128])
def test_collect_schema_neg_rejects_unsupported_integer_27565(
    dtype: pl.DataType,
) -> None:
    lf = pl.LazyFrame({"a": [1]}, schema={"a": dtype}).select(result=pl.col("a").neg())
    with pytest.raises(pl.exceptions.InvalidOperationError, match="neg"):
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
    for dtype in (pl.String, pl.Int64, pl.Float64):
        lf = pl.LazyFrame({"a": [None]}, schema={"a": dtype}).select(
            result=getattr(pl.col("a"), fn)()
        )
        assert lf.collect_schema()["result"] == pl.Float64


@pytest.mark.parametrize("fn", ["sqrt", "cbrt"])
@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (pl.Binary, b"1"),
        (pl.List(pl.Int64), [1]),
        (pl.Array(pl.Int64, 1), [1]),
        (pl.Struct({"x": pl.Int64}), {"x": 1}),
        (pl.Categorical, "a"),
        (pl.Map(pl.String, pl.Int64), {"a": 1}),
    ],
)
def test_collect_schema_sqrt_cbrt_rejects_unsupported_dtype_27565(
    fn: str, dtype: pl.DataType, value: object
) -> None:
    lf = pl.LazyFrame({"a": [value]}, schema={"a": dtype}).select(
        result=getattr(pl.col("a"), fn)()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match=fn):
        lf.collect_schema()


def test_collect_schema_trig_rejects_string_27565() -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": pl.String}).select(
        result=pl.col("a").sin()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="trigonometry"):
        lf.collect_schema()


@pytest.mark.parametrize("dtype", [pl.String, pl.Boolean, pl.Date, pl.Categorical])
def test_collect_schema_entropy_rejects_non_numeric_27565(
    dtype: pl.DataType,
) -> None:
    lf = pl.LazyFrame({"a": [None]}, schema={"a": dtype}).select(
        result=pl.col("a").entropy()
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="entropy"):
        lf.collect_schema()


@pytest.mark.parametrize(
    ("dtype", "values", "normalize", "expected"),
    [
        (pl.Float16, [0.25, 0.75], True, pl.Float16),
        (pl.Float32, [0.25, 0.75], True, pl.Float32),
        (pl.Int64, [1, 3], True, pl.Float64),
        (pl.Duration, [1, 3], True, pl.Float64),
        (pl.Duration, [1, 3], False, pl.Float64),
    ],
)
@pytest.mark.parametrize("grouped", [False, True])
def test_collect_schema_entropy_output_dtype_27565(
    dtype: pl.DataType,
    values: list[int | float],
    normalize: bool,
    expected: pl.DataType,
    *,
    grouped: bool,
) -> None:
    frame = pl.LazyFrame(
        {"group": [0, 0], "a": values}, schema={"group": pl.Int8, "a": dtype}
    )
    expr = pl.col("a").entropy(normalize=normalize)
    lf = (
        frame.group_by("group").agg(result=expr)
        if grouped
        else frame.select(result=expr)
    )

    schema = lf.collect_schema()
    in_memory = lf.collect(engine="in-memory")
    streaming = lf.collect(engine="streaming")
    assert schema["result"] == expected
    assert schema == in_memory.schema
    assert schema == streaming.schema
    if dtype == pl.Duration:
        assert streaming["result"].item() == pytest.approx(in_memory["result"].item())


def test_arr_get_oob_errors_at_schema_26088() -> None:
    lf = pl.LazyFrame({"arr": [[1, 2, 3]]}).cast({"arr": pl.Array(pl.Int64, shape=3)})

    with pytest.raises(pl.exceptions.ComputeError):
        lf.select(pl.col("arr").arr.get(5)).collect_schema()

    with pytest.raises(pl.exceptions.ComputeError):
        lf.select(pl.col("arr").arr.get(-4)).collect_schema()

    lf.select(pl.col("arr").arr.get(2)).collect_schema()

    lf.select(pl.col("arr").arr.get(-1)).collect_schema()

    lf.select(pl.col("arr").arr.get(5, null_on_oob=True)).collect_schema()
