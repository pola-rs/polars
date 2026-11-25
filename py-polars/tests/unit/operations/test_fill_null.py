from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import EngineType


def test_fill_null_minimal_upcast_4056() -> None:
    df = pl.DataFrame({"a": [-1, 2, None]})
    df = df.with_columns(pl.col("a").cast(pl.Int8))
    assert df.with_columns(pl.col(pl.Int8).fill_null(-1)).dtypes[0] == pl.Int8
    with pytest.deprecated_call():
        assert df.with_columns(pl.col(pl.Int8).fill_null(-1000)).dtypes[0] == pl.Int16


def test_fill_enum_upcast() -> None:
    dtype = pl.Enum(["a", "b"])
    s = pl.Series(["a", "b", None], dtype=dtype)
    s_filled = s.fill_null("b")
    expected = pl.Series(["a", "b", "b"], dtype=dtype)
    assert s_filled.dtype == dtype
    assert_series_equal(s_filled, expected)


def test_fill_null_enum_values() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    cases: list[tuple[list[str | None], str, list[str | None]]] = [
        (["a", None], "c", ["a", "c"]),
        (["a", None], "absent", ["a", None]),
        (["a", "b"], "absent", ["a", "b"]),
    ]
    for values, fill, expected in cases:
        s = pl.Series("s", values, dtype=dtype)
        assert_series_equal(s.fill_null(fill), pl.Series("s", expected, dtype=dtype))


@pytest.mark.parametrize(
    "make_categories",
    [
        pytest.param(pl.Categories.random, id="own"),
        pytest.param(lambda: pl.Categories(f"named-{uuid4()}"), id="named"),
        pytest.param(pl.Categories, id="global"),
        pytest.param(lambda: pl.Categories.random(physical=pl.UInt8), id="uint8"),
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_fill_null_categorical_adds_only_used_values(
    make_categories: Callable[[], pl.Categories], engine: EngineType
) -> None:
    cats = make_categories()
    dtype = pl.Categorical(cats)
    lf = pl.LazyFrame({"s": ["a", "b"]}, schema={"s": dtype})
    lf_null = pl.LazyFrame({"s": ["a", None]}, schema={"s": dtype})
    unused, used, other = (f"{name}-{uuid4()}" for name in ("unused", "used", "other"))

    result = lf.select(pl.col("s").fill_null(unused)).collect(engine=engine)
    assert_frame_equal(result, lf.collect())
    assert_series_equal(lf.collect()["s"].fill_null(unused), lf.collect()["s"])
    assert unused not in cats

    # Values in the output are added.
    result = lf_null.select(pl.col("s").fill_null(used)).collect(engine=engine)
    assert_frame_equal(result, pl.DataFrame({"s": ["a", used]}, schema={"s": dtype}))
    assert used in cats
    expr = pl.when(pl.col("s") == "a").then(pl.col("s")).otherwise(pl.lit(other))
    result = lf.select(expr).collect(engine=engine)
    assert_frame_equal(result, pl.DataFrame({"s": ["a", other]}, schema={"s": dtype}))
    assert other in cats


def test_fill_null_categorical_scalar_with_string_series() -> None:
    dtype = pl.Categorical(pl.Categories.random())
    result = pl.select(
        pl.lit(None, dtype=dtype).fill_null(pl.lit(pl.Series(["x", "y"])))
    )
    assert_series_equal(
        result.to_series(), pl.Series("literal", ["x", "y"], dtype=dtype)
    )


def test_fill_null_categorical_full_categories_raises() -> None:
    cats = pl.Categories.random(physical=pl.UInt8)
    s = pl.Series([f"c{i}" for i in range(255)] + [None], dtype=pl.Categorical(cats))
    with pytest.raises(ComputeError, match="more categories than the maximum"):
        s.fill_null("new")
    assert s.fill_null("c0")[-1] == "c0"


def test_fill_null_static_schema_4843() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": [1, None, 4],
        }
    ).lazy()

    df2 = df1.select([pl.col(pl.Int64).fill_null(0)])
    df3 = df2.select(pl.col(pl.Int64))
    assert df3.collect_schema() == {"a": pl.Int64, "b": pl.Int64}


def test_fill_null_non_lit() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([1, None], dtype=pl.Int32),
            "b": pl.Series([None, 2], dtype=pl.UInt32),
            "c": pl.Series([None, 2], dtype=pl.Int64),
            "d": pl.Series([None, 2], dtype=pl.Decimal(10, 2)),
        }
    )
    assert df.fill_null(0).select(pl.all().null_count()).transpose().sum().item() == 0


def test_fill_null_f32_with_lit() -> None:
    # ensure the literal integer does not upcast the f32 to an f64
    df = pl.DataFrame({"a": [1.1, 1.2]}, schema=[("a", pl.Float32)])
    assert df.fill_null(value=0).dtypes == [pl.Float32]


def test_fill_null_lit_() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([1, None], dtype=pl.Int32),
            "b": pl.Series([None, 2], dtype=pl.UInt32),
            "c": pl.Series([None, 2], dtype=pl.Int64),
        }
    )
    assert (
        df.fill_null(pl.lit(0)).select(pl.all().null_count()).transpose().sum().item()
        == 0
    )


def test_fill_null_decimal_with_int_14331() -> None:
    s = pl.Series("a", ["1.1", None], dtype=pl.Decimal(precision=None, scale=5))
    result = s.fill_null(0)
    expected = pl.Series("a", ["1.1", "0.0"], dtype=pl.Decimal(precision=None, scale=5))
    assert_series_equal(result, expected)


def test_fill_null_date_with_int_11362() -> None:
    match = "got invalid or ambiguous dtypes"

    s = pl.Series([datetime.date(2000, 1, 1)])
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        s.fill_null(0)

    s = pl.Series([None], dtype=pl.Date)
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        s.fill_null(1)


def test_fill_null_int_dtype_15546() -> None:
    df = pl.Series("a", [1, 2, None], dtype=pl.Int8).to_frame().lazy()
    result = df.fill_null(0).collect()
    expected = pl.Series("a", [1, 2, 0], dtype=pl.Int8).to_frame()
    assert_frame_equal(result, expected)


def test_fill_null_with_list_10869() -> None:
    assert_series_equal(
        pl.Series([[1], None]).fill_null([2]),
        pl.Series([[1], [2]]),
    )

    match = "failed to determine supertype"
    with pytest.raises(pl.exceptions.SchemaError, match=match):
        pl.Series([1, None]).fill_null([2])


def test_unequal_lengths_22018() -> None:
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([1, None]).fill_null(pl.Series([1] * 3))
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([1, 2]).fill_null(pl.Series([1] * 3))


def test_self_broadcast() -> None:
    assert_series_equal(
        pl.Series([1]).fill_null(pl.Series(range(3))),
        pl.Series([1] * 3),
    )

    assert_series_equal(
        pl.Series([None]).fill_null(pl.Series(range(3))),
        pl.Series(range(3)),
    )


def test_forward_fill_chunking_25273() -> None:
    df = pl.DataFrame(
        {
            "key": [0, 1, 1],
            "a": [None, None, 0],
        }
    )

    df = df.with_columns(a=pl.concat([pl.col.a.head(1), pl.col.a.tail(2)]))
    df = df.select(pl.col.a.filter(pl.col.key == 1))
    df = df.with_columns(ff=pl.col.a.forward_fill())
    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "a": [None, 0],
                "ff": [None, 0],
            }
        ),
    )


def test_forward_fill_is_length_preserving() -> None:
    assert_series_equal(
        pl.Series([[1]]).list.agg(pl.element().first().forward_fill()),
        pl.Series([1]),
    )


@pytest.mark.slow
@given(
    s=series(allow_null=True),
    limit=st.one_of(st.none(), st.integers(min_value=0, max_value=10)),
)
@pytest.mark.parametrize("fill", ["forward_fill", "backward_fill"])
def test_fill_streaming_matches_in_memory(
    fill: str, s: pl.Series, limit: int | None
) -> None:
    q = pl.LazyFrame({"a": s}).select(getattr(pl.col("a"), fill)(limit=limit))
    expected = q.collect(engine="in-memory")
    result = q.collect(engine="streaming")
    assert_series_equal(result["a"], expected["a"])


def test_fill_null_null_dtype_24451() -> None:
    # Test that fill_null changes Null dtype to fill value's dtype and fills values
    df = pl.DataFrame({"col1": [None, None, None], "col2": [None, None, None]})

    result = df.fill_null("rabbit")
    assert result.dtypes == [pl.String, pl.String]
    # Values are filled with the fill value
    assert result["col1"].to_list() == ["rabbit", "rabbit", "rabbit"]
    assert result["col2"].to_list() == ["rabbit", "rabbit", "rabbit"]


def test_fill_null_arr_nonull_cat_28247() -> None:
    s = pl.Series([["a", "b"]], dtype=pl.Array(pl.Categorical, 2))
    assert_series_equal(s, s.arr.eval(pl.element().fill_null(strategy="forward")))


def test_fill_null_cast_to_input_dtype_31073() -> None:
    s = pl.Series("a", [datetime.datetime(2020, 1, 1), None])
    s = s.dt.replace_time_zone("Europe/Amsterdam")
    fill = pl.lit(datetime.datetime(2000, 1, 1)).dt.replace_time_zone("UTC")
    expected = pl.Series(
        "a", [datetime.datetime(2020, 1, 1), datetime.datetime(2000, 1, 1, 1)]
    ).dt.replace_time_zone("Europe/Amsterdam")
    assert_series_equal(s.fill_null(fill), expected)

    s = pl.Series("a", [Decimal("1.50"), None], dtype=pl.Decimal(10, 2))
    expected = pl.Series(
        "a", [Decimal("1.50"), Decimal("7.00")], dtype=pl.Decimal(10, 2)
    )
    assert_series_equal(s.fill_null(7), expected)


def test_fill_null_supertype_deprecated_31073() -> None:
    s = pl.Series("a", [1, None], dtype=pl.Int8)
    with pytest.deprecated_call(match="will raise an error in 3.0"):
        out = s.fill_null(1.5)
    assert_series_equal(out, pl.Series("a", [1.0, 1.5], dtype=pl.Float64))


def test_fill_null_no_supertype_31073() -> None:
    s = pl.Series("a", [1, None])
    with pytest.raises(
        pl.exceptions.SchemaError, match="failed to determine supertype"
    ):
        s.fill_null(b"x")


def test_fill_null_float_literal_keeps_float32_31073(
    recwarn: pytest.WarningsRecorder,
) -> None:
    s = pl.Series("a", [1.5, None], dtype=pl.Float32)
    expected = pl.Series("a", [1.5, 0.1], dtype=pl.Float32)
    assert_series_equal(s.fill_null(0.1), expected)
    assert len(recwarn) == 0


def test_fill_null_datetime_literal_time_unit_31073(
    recwarn: pytest.WarningsRecorder,
) -> None:
    fill = datetime.datetime(2021, 1, 1, 12)
    s = pl.Series("a", [datetime.datetime(2020, 1, 1), None], dtype=pl.Datetime("ns"))
    expected = pl.Series(
        "a", [datetime.datetime(2020, 1, 1), fill], dtype=pl.Datetime("ns")
    )
    assert_series_equal(s.fill_null(fill), expected)
    assert len(recwarn) == 0

    s = pl.Series("a", [datetime.datetime(2020, 1, 1), None], dtype=pl.Datetime("ms"))
    with pytest.deprecated_call(match="will raise an error in 3.0"):
        out = s.fill_null(fill)
    expected = pl.Series(
        "a", [datetime.datetime(2020, 1, 1), fill], dtype=pl.Datetime("ms")
    )
    assert_series_equal(out, expected)


@pytest.mark.parametrize(("fill", "dtype"), [(1, pl.Int32), ("x", pl.String)])
def test_null_input_takes_fill_dtype_31073(
    fill: int | str, dtype: pl.DataType, recwarn: pytest.WarningsRecorder
) -> None:
    out = pl.LazyFrame({"a": [None, None]}).select(
        pl.col("a").fill_null(fill).alias("fill_null"),
        pl.col("a").shift(1, fill_value=fill).alias("shift"),
    )
    expected = pl.DataFrame(
        {"fill_null": [fill, fill], "shift": [fill, None]},
        schema={"fill_null": dtype, "shift": dtype},
    )
    assert_frame_equal(out.collect(), expected)

    out = pl.LazyFrame().select(
        pl.lit(None).fill_null(fill).alias("fill_null"),
        pl.lit(None).shift(1, fill_value=fill).alias("shift"),
    )
    expected = pl.DataFrame(
        {"fill_null": [fill], "shift": [fill]},
        schema={"fill_null": dtype, "shift": dtype},
    )
    assert_frame_equal(out.collect(), expected)
    assert len(recwarn) == 0
