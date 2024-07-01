import operator
from collections import OrderedDict
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars import (
    Date,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from polars.exceptions import ColumnNotFoundError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import INTEGER_DTYPES, NUMERIC_DTYPES


def test_sqrt_neg_inf() -> None:
    out = pl.DataFrame(
        {
            "val": [float("-Inf"), -9, 0, 9, float("Inf")],
        }
    ).with_columns(pl.col("val").sqrt().alias("sqrt"))
    # comparing nans and infinities by string value as they are not cmp
    assert str(out["sqrt"].to_list()) == str(
        [float("nan"), float("nan"), 0.0, 3.0, float("Inf")]
    )


def test_arithmetic_with_logical_on_series_4920() -> None:
    assert (pl.Series([date(2022, 6, 3)]) - date(2022, 1, 1)).dtype == pl.Duration("ms")


@pytest.mark.parametrize(
    ("left", "right", "expected_value", "expected_dtype"),
    [
        (date(2021, 1, 1), date(2020, 1, 1), timedelta(days=366), pl.Duration("ms")),
        (
            datetime(2021, 1, 1),
            datetime(2020, 1, 1),
            timedelta(days=366),
            pl.Duration("us"),
        ),
        (timedelta(days=1), timedelta(days=2), timedelta(days=-1), pl.Duration("us")),
        (2.0, 3.0, -1.0, pl.Float64),
    ],
)
def test_arithmetic_sub(
    left: object, right: object, expected_value: object, expected_dtype: pl.DataType
) -> None:
    result = left - pl.Series([right])
    expected = pl.Series("", [expected_value], dtype=expected_dtype)
    assert_series_equal(result, expected)
    result = pl.Series([left]) - right
    assert_series_equal(result, expected)


def test_struct_arithmetic() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    ).select(pl.cum_sum_horizontal("a", "c"))
    assert df.select(pl.col("cum_sum") * 2).to_dict(as_series=False) == {
        "cum_sum": [{"a": 2, "c": 12}, {"a": 4, "c": 16}]
    }
    assert df.select(pl.col("cum_sum") - 2).to_dict(as_series=False) == {
        "cum_sum": [{"a": -1, "c": 4}, {"a": 0, "c": 6}]
    }
    assert df.select(pl.col("cum_sum") + 2).to_dict(as_series=False) == {
        "cum_sum": [{"a": 3, "c": 8}, {"a": 4, "c": 10}]
    }
    assert df.select(pl.col("cum_sum") / 2).to_dict(as_series=False) == {
        "cum_sum": [{"a": 0.5, "c": 3.0}, {"a": 1.0, "c": 4.0}]
    }
    assert df.select(pl.col("cum_sum") // 2).to_dict(as_series=False) == {
        "cum_sum": [{"a": 0, "c": 3}, {"a": 1, "c": 4}]
    }

    # inline, this checks cum_sum reports the right output type
    assert pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).select(
        pl.cum_sum_horizontal("a", "c") * 3
    ).to_dict(as_series=False) == {"cum_sum": [{"a": 3, "c": 18}, {"a": 6, "c": 24}]}


def test_simd_float_sum_determinism() -> None:
    out = []
    for _ in range(10):
        a = pl.Series(
            [
                0.021415853782953836,
                0.06234123511682772,
                0.016962384922753124,
                0.002595968402539279,
                0.007632765529696731,
                0.012105848332077212,
                0.021439787151032317,
                0.3223049133700719,
                0.10526670729539435,
                0.0859029285522487,
            ]
        )
        out.append(a.sum())

    assert out == [
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
        0.6579683924555951,
    ]


def test_floor_division_float_int_consistency() -> None:
    a = np.random.randn(10) * 10

    assert (pl.Series(a) // 5).to_list() == list(a // 5)
    assert (pl.Series(a, dtype=pl.Int32) // 5).to_list() == list(
        (a.astype(int) // 5).astype(int)
    )


def test_series_expr_arithm() -> None:
    s = pl.Series([1, 2, 3])
    assert (s + pl.col("a")).meta == pl.lit(s) + pl.col("a")
    assert (s - pl.col("a")).meta == pl.lit(s) - pl.col("a")
    assert (s / pl.col("a")).meta == pl.lit(s) / pl.col("a")
    assert (s // pl.col("a")).meta == pl.lit(s) // pl.col("a")
    assert (s * pl.col("a")).meta == pl.lit(s) * pl.col("a")
    assert (s % pl.col("a")).meta == pl.lit(s) % pl.col("a")


def test_fused_arithm() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [5, 5, 5],
        }
    )

    q = df.lazy().select(
        pl.col("a") * pl.col("b") + pl.col("c"),
        (pl.col("a") + pl.col("b") * pl.col("c")).alias("2"),
    )
    # the extra aliases are because the fma does operation reordering
    assert (
        """col("c").fma([col("a"), col("b")]).alias("a"), col("a").fma([col("b"), col("c")]).alias("2")"""
        in q.explain()
    )
    assert q.collect().to_dict(as_series=False) == {
        "a": [15, 45, 95],
        "2": [51, 102, 153],
    }
    # fsm
    q = df.lazy().select(pl.col("a") - pl.col("b") * pl.col("c"))
    assert """col("a").fsm([col("b"), col("c")])""" in q.explain()
    assert q.collect()["a"].to_list() == [-49, -98, -147]
    # fms
    q = df.lazy().select(pl.col("a") * pl.col("b") - pl.col("c"))
    assert """col("a").fms([col("b"), col("c")])""" in q.explain()
    assert q.collect()["a"].to_list() == [5, 35, 85]

    # check if we constant fold instead of fma
    q = df.lazy().select(pl.lit(1) * pl.lit(2) - pl.col("c"))
    assert """(2) - (col("c")""" in q.explain()

    # Check if fused is turned off for literals see: #9857
    for expr in [
        pl.col("c") * 2 + 5,
        pl.col("c") * 2 + pl.col("c"),
        pl.col("c") * 2 - 5,
        pl.col("c") * 2 - pl.col("c"),
        5 - pl.col("c") * 2,
        pl.col("c") - pl.col("c") * 2,
    ]:
        q = df.lazy().select(expr)
        assert all(
            el not in q.explain() for el in ["fms", "fsm", "fma"]
        ), f"Fused Arithmetic applied on literal {expr}: {q.explain()}"


def test_literal_no_upcast() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Float32)})

    q = (
        df.lazy()
        .select(
            (pl.col("a") * -5 + 2).alias("fma"),
            (2 - pl.col("a") * 5).alias("fsm"),
            (pl.col("a") * 5 - 2).alias("fms"),
        )
        .collect()
    )
    assert set(q.schema.values()) == {
        pl.Float32
    }, "Literal * Column (Float32) should not lead upcast"


def test_boolean_addition() -> None:
    s = pl.DataFrame(
        {"a": [True, False, False], "b": [True, False, True]}
    ).sum_horizontal()

    assert s.dtype == pl.get_index_type()
    assert s.to_list() == [2, 0, 1]
    df = pl.DataFrame(
        {"a": [True], "b": [False]},
    ).select(pl.sum_horizontal("a", "b"))
    assert df.dtypes == [pl.get_index_type()]


def test_bitwise_6311() -> None:
    df = pl.DataFrame({"col1": [0, 1, 2, 3], "flag": [0, 0, 0, 0]})

    assert (
        df.with_columns(
            pl.when((pl.col("col1") < 1) | (pl.col("col1") >= 3))
            .then(pl.col("flag") | 2)  # set flag b0010
            .otherwise(pl.col("flag"))
        ).with_columns(
            pl.when(pl.col("col1") > -1)
            .then(pl.col("flag") | 4)
            .otherwise(pl.col("flag"))
        )
    ).to_dict(as_series=False) == {"col1": [0, 1, 2, 3], "flag": [6, 4, 4, 6]}


def test_arithmetic_null_count() -> None:
    df = pl.DataFrame({"a": [1, None, 2], "b": [None, 2, 1]})
    out = df.select(
        no_broadcast=pl.col("a") + pl.col("b"),
        broadcast_left=1 + pl.col("b"),
        broadcast_right=pl.col("a") + 1,
    )
    assert out.null_count().to_dict(as_series=False) == {
        "no_broadcast": [2],
        "broadcast_left": [1],
        "broadcast_right": [1],
    }


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.floordiv,
        operator.mod,
        operator.mul,
        operator.sub,
    ],
)
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_operator_arithmetic_with_nulls(op: Any, dtype: pl.DataType) -> None:
    df = pl.DataFrame({"n": [2, 3]}, schema={"n": dtype})
    s = df.to_series()

    df_expected = pl.DataFrame({"n": [None, None]}, schema={"n": dtype})
    s_expected = df_expected.to_series()

    # validate expr, frame, and series behaviour with null value arithmetic
    op_name = op.__name__
    for null_expr in (None, pl.lit(None)):
        assert_frame_equal(df_expected, df.select(op(pl.col("n"), null_expr)))
        assert_frame_equal(
            df_expected, df.select(getattr(pl.col("n"), op_name)(null_expr))
        )

    assert_frame_equal(df_expected, op(df, None))
    assert_series_equal(s_expected, op(s, None))


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mod,
        operator.mul,
        operator.sub,
    ],
)
def test_null_column_arithmetic(op: Any) -> None:
    df = pl.DataFrame({"a": [None, None], "b": [None, None]})
    expected_df = pl.DataFrame({"a": [None, None]})

    output_df = df.select(op(pl.col("a"), pl.col("b")))
    assert_frame_equal(expected_df, output_df)
    # test broadcast right
    output_df = df.select(op(pl.col("a"), pl.Series([None])))
    assert_frame_equal(expected_df, output_df)
    # test broadcast left
    output_df = df.select(op(pl.Series("a", [None]), pl.col("a")))
    assert_frame_equal(expected_df, output_df)


def test_bool_floordiv() -> None:
    df = pl.DataFrame({"x": [True]})

    with pytest.raises(
        InvalidOperationError,
        match="floor_div operation not supported for dtype `bool`",
    ):
        df.with_columns(pl.col("x").floordiv(2))


def test_arithmetic_in_aggregation_3739() -> None:
    def demean_dot() -> pl.Expr:
        x = pl.col("x")
        y = pl.col("y")
        x1 = x - x.mean()
        y1 = y - y.mean()
        return (x1 * y1).sum().alias("demean_dot")

    assert (
        pl.DataFrame(
            {
                "key": ["a", "a", "a", "a"],
                "x": [4, 2, 2, 4],
                "y": [2, 0, 2, 0],
            }
        )
        .group_by("key")
        .agg(
            [
                demean_dot(),
            ]
        )
    ).to_dict(as_series=False) == {"key": ["a"], "demean_dot": [0.0]}


def test_arithmetic_on_df() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    for df_mul in (df * 2, 2 * df):
        expected = pl.DataFrame({"a": [2.0, 4.0], "b": [6.0, 8.0]})
        assert_frame_equal(df_mul, expected)

    for df_plus in (df + 2, 2 + df):
        expected = pl.DataFrame({"a": [3.0, 4.0], "b": [5.0, 6.0]})
        assert_frame_equal(df_plus, expected)

    df_div = df / 2
    expected = pl.DataFrame({"a": [0.5, 1.0], "b": [1.5, 2.0]})
    assert_frame_equal(df_div, expected)

    df_minus = df - 2
    expected = pl.DataFrame({"a": [-1.0, 0.0], "b": [1.0, 2.0]})
    assert_frame_equal(df_minus, expected)

    df_mod = df % 2
    expected = pl.DataFrame({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert_frame_equal(df_mod, expected)

    df2 = pl.DataFrame({"c": [10]})

    out = df + df2
    expected = pl.DataFrame({"a": [11.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df - df2
    expected = pl.DataFrame({"a": [-9.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df / df2
    expected = pl.DataFrame({"a": [0.1, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df * df2
    expected = pl.DataFrame({"a": [10.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df % df2
    expected = pl.DataFrame({"a": [1.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    # cannot do arithmetic with a sequence
    with pytest.raises(TypeError, match="operation not supported"):
        _ = df + [1]  # type: ignore[operator]


def test_df_series_division() -> None:
    df = pl.DataFrame(
        {
            "a": [2, 2, 4, 4, 6, 6],
            "b": [2, 2, 10, 5, 6, 6],
        }
    )
    s = pl.Series([2, 2, 2, 2, 2, 2])
    assert (df / s).to_dict(as_series=False) == {
        "a": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "b": [1.0, 1.0, 5.0, 2.5, 3.0, 3.0],
    }
    assert (df // s).to_dict(as_series=False) == {
        "a": [1, 1, 2, 2, 3, 3],
        "b": [1, 1, 5, 2, 3, 3],
    }


@pytest.mark.parametrize(
    "s", [pl.Series([1, 2], dtype=Int64), pl.Series([1, 2], dtype=Float64)]
)
def test_arithmetic_series(s: pl.Series) -> None:
    a = s
    b = s

    assert ((a * b) == [1, 4]).sum() == 2
    assert ((a / b) == [1.0, 1.0]).sum() == 2
    assert ((a + b) == [2, 4]).sum() == 2
    assert ((a - b) == [0, 0]).sum() == 2
    assert ((a + 1) == [2, 3]).sum() == 2
    assert ((a - 1) == [0, 1]).sum() == 2
    assert ((a / 1) == [1.0, 2.0]).sum() == 2
    assert ((a // 2) == [0, 1]).sum() == 2
    assert ((a * 2) == [2, 4]).sum() == 2
    assert ((2 + a) == [3, 4]).sum() == 2
    assert ((1 - a) == [0, -1]).sum() == 2
    assert ((2 * a) == [2, 4]).sum() == 2

    # integer division
    assert_series_equal(1 / a, pl.Series([1.0, 0.5]))
    expected = pl.Series([1, 0]) if s.dtype == Int64 else pl.Series([1.0, 0.5])
    assert_series_equal(1 // a, expected)
    # modulo
    assert ((1 % a) == [0, 1]).sum() == 2
    assert ((a % 1) == [0, 0]).sum() == 2
    # negate
    assert (-a == [-1, -2]).sum() == 2
    # unary plus
    assert (+a == a).all()
    # wrong dtypes in rhs operands
    assert ((1.0 - a) == [0.0, -1.0]).sum() == 2
    assert ((1.0 / a) == [1.0, 0.5]).sum() == 2
    assert ((1.0 * a) == [1, 2]).sum() == 2
    assert ((1.0 + a) == [2, 3]).sum() == 2
    assert ((1.0 % a) == [0, 1]).sum() == 2


def test_arithmetic_datetime() -> None:
    a = pl.Series("a", [datetime(2021, 1, 1)])
    with pytest.raises(TypeError):
        a // 2
    with pytest.raises(TypeError):
        a / 2
    with pytest.raises(TypeError):
        a * 2
    with pytest.raises(TypeError):
        a % 2
    with pytest.raises(
        InvalidOperationError,
    ):
        a**2
    with pytest.raises(TypeError):
        2 / a
    with pytest.raises(TypeError):
        2 // a
    with pytest.raises(TypeError):
        2 * a
    with pytest.raises(TypeError):
        2 % a
    with pytest.raises(
        InvalidOperationError,
    ):
        2**a


def test_power_series() -> None:
    a = pl.Series([1, 2], dtype=Int64)
    b = pl.Series([None, 2.0], dtype=Float64)
    c = pl.Series([date(2020, 2, 28), date(2020, 3, 1)], dtype=Date)
    d = pl.Series([1, 2], dtype=UInt8)
    e = pl.Series([1, 2], dtype=Int8)
    f = pl.Series([1, 2], dtype=UInt16)
    g = pl.Series([1, 2], dtype=Int16)
    h = pl.Series([1, 2], dtype=UInt32)
    i = pl.Series([1, 2], dtype=Int32)
    j = pl.Series([1, 2], dtype=UInt64)
    k = pl.Series([1, 2], dtype=Int64)
    m = pl.Series([2**33, 2**33], dtype=UInt64)

    # pow
    assert_series_equal(a**2, pl.Series([1, 4], dtype=Int64))
    assert_series_equal(b**3, pl.Series([None, 8.0], dtype=Float64))
    assert_series_equal(a**a, pl.Series([1, 4], dtype=Int64))
    assert_series_equal(b**b, pl.Series([None, 4.0], dtype=Float64))
    assert_series_equal(a**b, pl.Series([None, 4.0], dtype=Float64))
    assert_series_equal(d**d, pl.Series([1, 4], dtype=UInt8))
    assert_series_equal(e**d, pl.Series([1, 4], dtype=Int8))
    assert_series_equal(f**d, pl.Series([1, 4], dtype=UInt16))
    assert_series_equal(g**d, pl.Series([1, 4], dtype=Int16))
    assert_series_equal(h**d, pl.Series([1, 4], dtype=UInt32))
    assert_series_equal(i**d, pl.Series([1, 4], dtype=Int32))
    assert_series_equal(j**d, pl.Series([1, 4], dtype=UInt64))
    assert_series_equal(k**d, pl.Series([1, 4], dtype=Int64))

    with pytest.raises(
        InvalidOperationError,
        match="`pow` operation not supported for dtype `null` as exponent",
    ):
        a ** pl.lit(None)

    with pytest.raises(
        InvalidOperationError,
        match="`pow` operation not supported for dtype `date` as base",
    ):
        c**2
    with pytest.raises(
        InvalidOperationError,
        match="`pow` operation not supported for dtype `date` as exponent",
    ):
        2**c

    with pytest.raises(ColumnNotFoundError):
        a ** "hi"  # type: ignore[operator]

    # Raising to UInt64: raises if can't be downcast safely to UInt32...
    with pytest.raises(
        InvalidOperationError, match="conversion from `u64` to `u32` failed"
    ):
        a**m
    # ... but succeeds otherwise.
    assert_series_equal(a**j, pl.Series([1, 4], dtype=Int64))

    # rpow
    assert_series_equal(2.0**a, pl.Series("literal", [2.0, 4.0], dtype=Float64))
    assert_series_equal(2**b, pl.Series("literal", [None, 4.0], dtype=Float64))

    with pytest.raises(ColumnNotFoundError):
        "hi" ** a

    # Series.pow() method
    assert_series_equal(a.pow(2), pl.Series([1, 4], dtype=Int64))


@pytest.mark.parametrize(
    ("expected", "expr"),
    [
        (
            np.array([[2, 4], [6, 8]]),
            pl.col("a") + pl.col("a"),
        ),
        (
            np.array([[0, 0], [0, 0]]),
            pl.col("a") - pl.col("a"),
        ),
        (
            np.array([[1, 4], [9, 16]]),
            pl.col("a") * pl.col("a"),
        ),
        (
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            pl.col("a") / pl.col("a"),
        ),
    ],
)
def test_array_arithmetic_same_size(expected: Any, expr: pl.Expr) -> None:
    df = pl.Series("a", np.array([[1, 2], [3, 4]])).to_frame()

    assert_frame_equal(
        df.select(expr),
        pl.Series("a", expected).to_frame(),
    )


def test_schema_owned_arithmetic_5669() -> None:
    df = (
        pl.LazyFrame({"A": [1, 2, 3]})
        .filter(pl.col("A") >= 3)
        .with_columns(-pl.col("A").alias("B"))
        .collect()
    )
    assert df.columns == ["A", "B"]
    assert df.rows() == [(3, -3)]


def test_schema_true_divide_6643() -> None:
    df = pl.DataFrame({"a": [1]})
    a = pl.col("a")
    assert df.lazy().select(a / 2).select(pl.col(pl.Int64)).collect().shape == (0, 0)


def test_literal_subtract_schema_13284() -> None:
    assert (
        pl.LazyFrame({"a": [23, 30]}, schema={"a": pl.UInt8})
        .with_columns(pl.col("a") - pl.lit(1))
        .group_by("a")
        .len()
    ).collect_schema() == OrderedDict([("a", pl.UInt8), ("len", pl.UInt32)])


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_int_operator_stability(dtype: pl.DataType) -> None:
    s = pl.Series(values=[10], dtype=dtype)
    assert pl.select(pl.lit(s) // 2).dtypes == [dtype]
    assert pl.select(pl.lit(s) + 2).dtypes == [dtype]
    assert pl.select(pl.lit(s) - 2).dtypes == [dtype]
    assert pl.select(pl.lit(s) * 2).dtypes == [dtype]
    assert pl.select(pl.lit(s) / 2).dtypes == [pl.Float64]


def test_duration_division_schema() -> None:
    df = pl.DataFrame({"a": [1]})
    q = (
        df.lazy()
        .with_columns(pl.col("a").cast(pl.Duration))
        .select(pl.col("a") / pl.col("a"))
    )

    assert q.collect_schema() == {"a": pl.Float64}
    assert q.collect().to_dict(as_series=False) == {"a": [1.0]}


@pytest.mark.parametrize(
    ("a", "b", "op"),
    [
        (pl.Duration, pl.Int32, "+"),
        (pl.Int32, pl.Duration, "+"),
        (pl.Time, pl.Int32, "+"),
        (pl.Int32, pl.Time, "+"),
        (pl.Date, pl.Int32, "+"),
        (pl.Int32, pl.Date, "+"),
        (pl.Datetime, pl.Duration, "*"),
        (pl.Duration, pl.Datetime, "*"),
        (pl.Date, pl.Duration, "*"),
        (pl.Duration, pl.Date, "*"),
        (pl.Time, pl.Duration, "*"),
        (pl.Duration, pl.Time, "*"),
    ],
)
def test_raise_invalid_temporal(a: pl.DataType, b: pl.DataType, op: str) -> None:
    a = pl.Series("a", [], dtype=a)  # type: ignore[assignment]
    b = pl.Series("b", [], dtype=b)  # type: ignore[assignment]
    _df = pl.DataFrame([a, b])

    with pytest.raises(InvalidOperationError):
        eval(f"_df.select(pl.col('a') {op} pl.col('b'))")


def test_arithmetic_duration_div_multiply() -> None:
    df = pl.DataFrame([pl.Series("a", [100, 200, 3000], dtype=pl.Duration)])

    q = df.lazy().with_columns(
        b=pl.col("a") / 2,
        c=pl.col("a") / 2.5,
        d=pl.col("a") * 2,
        e=pl.col("a") * 2.5,
        f=pl.col("a") / pl.col("a"),  # a constant float
    )
    assert q.collect_schema() == pl.Schema(
        [
            ("a", pl.Duration(time_unit="us")),
            ("b", pl.Duration(time_unit="us")),
            ("c", pl.Duration(time_unit="us")),
            ("d", pl.Unknown()),
            ("e", pl.Unknown()),
            ("f", pl.Float64()),
        ]
    )
    assert q.collect().to_dict(as_series=False) == {
        "a": [
            timedelta(microseconds=100),
            timedelta(microseconds=200),
            timedelta(microseconds=3000),
        ],
        "b": [
            timedelta(microseconds=50),
            timedelta(microseconds=100),
            timedelta(microseconds=1500),
        ],
        "c": [
            timedelta(microseconds=40),
            timedelta(microseconds=80),
            timedelta(microseconds=1200),
        ],
        "d": [
            timedelta(microseconds=200),
            timedelta(microseconds=400),
            timedelta(microseconds=6000),
        ],
        "e": [
            timedelta(microseconds=250),
            timedelta(microseconds=500),
            timedelta(microseconds=7500),
        ],
        "f": [1.0, 1.0, 1.0],
    }

    # rhs

    q = df.lazy().with_columns(
        b=2 * pl.col("a"),
        c=2.5 * pl.col("a"),
    )
    assert q.collect_schema() == pl.Schema(
        [
            ("a", pl.Duration(time_unit="us")),
            ("b", pl.Duration(time_unit="us")),
            ("c", pl.Duration(time_unit="us")),
        ]
    )
    assert q.collect().to_dict(as_series=False) == {
        "a": [
            timedelta(microseconds=100),
            timedelta(microseconds=200),
            timedelta(microseconds=3000),
        ],
        "b": [
            timedelta(microseconds=200),
            timedelta(microseconds=400),
            timedelta(microseconds=6000),
        ],
        "c": [
            timedelta(microseconds=250),
            timedelta(microseconds=500),
            timedelta(microseconds=7500),
        ],
    }


def test_invalid_shapes_err() -> None:
    with pytest.raises(
        InvalidOperationError,
        match=r"cannot do arithmetic operation on series of different lengths: got 2 and 3",
    ):
        pl.Series([1, 2]) + pl.Series([1, 2, 3])


def test_date_datetime_sub() -> None:
    df = pl.DataFrame({"foo": [date(2020, 1, 1)], "bar": [datetime(2020, 1, 5)]})

    assert df.select(
        pl.col("foo") - pl.col("bar"),
        pl.col("bar") - pl.col("foo"),
    ).to_dict(as_series=False) == {
        "foo": [timedelta(days=-4)],
        "bar": [timedelta(days=4)],
    }


def test_raise_invalid_shape() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.DataFrame([[1, 2], [3, 4]]) * pl.DataFrame([1, 2, 3])
