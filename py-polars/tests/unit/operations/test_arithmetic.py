from datetime import date, datetime, timedelta

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_sqrt_neg_inf() -> None:
    out = pl.DataFrame(
        {
            "val": [float("-Inf"), -9, 0, 9, float("Inf")],
        }
    ).with_columns(pl.col("val").sqrt().alias("sqrt"))
    # comparing nans and infinities by string value as they are not cmp
    assert str(out["sqrt"].to_list()) == str(
        [float("NaN"), float("NaN"), 0.0, 3.0, float("Inf")]
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
    ).select(pl.cumsum_horizontal("a", "c"))
    assert df.select(pl.col("cumsum") * 2).to_dict(as_series=False) == {
        "cumsum": [{"a": 2, "c": 12}, {"a": 4, "c": 16}]
    }
    assert df.select(pl.col("cumsum") - 2).to_dict(as_series=False) == {
        "cumsum": [{"a": -1, "c": 4}, {"a": 0, "c": 6}]
    }
    assert df.select(pl.col("cumsum") + 2).to_dict(as_series=False) == {
        "cumsum": [{"a": 3, "c": 8}, {"a": 4, "c": 10}]
    }
    assert df.select(pl.col("cumsum") / 2).to_dict(as_series=False) == {
        "cumsum": [{"a": 0.5, "c": 3.0}, {"a": 1.0, "c": 4.0}]
    }
    assert df.select(pl.col("cumsum") // 2).to_dict(as_series=False) == {
        "cumsum": [{"a": 0, "c": 3}, {"a": 1, "c": 4}]
    }

    # inline, this check cumsum reports the right output type
    assert pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).select(
        pl.cumsum_horizontal("a", "c") * 3
    ).to_dict(as_series=False) == {"cumsum": [{"a": 3, "c": 18}, {"a": 6, "c": 24}]}


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


def test_unary_plus() -> None:
    data = [1, 2]
    df = pl.DataFrame({"x": data})
    assert df.select(+pl.col("x"))[:, 0].to_list() == data

    with pytest.raises(pl.exceptions.ComputeError):
        pl.select(+pl.lit(""))


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

    assert s.dtype == pl.utils.get_index_type()
    assert s.to_list() == [2, 0, 1]
    df = pl.DataFrame(
        {"a": [True], "b": [False]},
    ).select(pl.sum_horizontal("a", "b"))
    assert df.dtypes == [pl.utils.get_index_type()]


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
