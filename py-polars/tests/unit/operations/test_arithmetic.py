import typing
from datetime import date

import numpy as np
import pytest

import polars as pl


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


def test_struct_arithmetic() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    ).select(pl.cumsum(["a", "c"]))
    assert df.select(pl.col("cumsum") * 2).to_dict(False) == {
        "cumsum": [{"a": 2, "c": 12}, {"a": 4, "c": 16}]
    }
    assert df.select(pl.col("cumsum") - 2).to_dict(False) == {
        "cumsum": [{"a": -1, "c": 4}, {"a": 0, "c": 6}]
    }
    assert df.select(pl.col("cumsum") + 2).to_dict(False) == {
        "cumsum": [{"a": 3, "c": 8}, {"a": 4, "c": 10}]
    }
    assert df.select(pl.col("cumsum") / 2).to_dict(False) == {
        "cumsum": [{"a": 0.5, "c": 3.0}, {"a": 1.0, "c": 4.0}]
    }
    assert df.select(pl.col("cumsum") // 2).to_dict(False) == {
        "cumsum": [{"a": 0, "c": 3}, {"a": 1, "c": 4}]
    }

    # inline, this check cumsum reports the right output type
    assert pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).select(
        pl.cumsum(["a", "c"]) * 3
    ).to_dict(False) == {"cumsum": [{"a": 3, "c": 18}, {"a": 6, "c": 24}]}


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


@typing.no_type_check
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
