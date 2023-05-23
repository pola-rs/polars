from datetime import datetime
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_series_equal


@pytest.mark.parametrize(
    ("value", "n", "dtype", "expected_dtype"),
    [
        (2**31, 5, None, pl.Int64),
        (2**31 - 1, 5, None, pl.Int32),
        (-(2**31) - 1, 3, None, pl.Int64),
        (-(2**31), 3, None, pl.Int32),
        ("foo", 2, None, pl.Utf8),
        (1.0, 5, None, pl.Float64),
        (True, 4, None, pl.Boolean),
        (None, 7, None, pl.Null),
        (0, 0, None, pl.Int32),
        (8, 2, pl.UInt8, pl.UInt8),
        pytest.param(
            datetime(2023, 2, 2),
            3,
            None,
            pl.Datetime,
            marks=pytest.mark.skip("Not implemented properly yet for lazy"),
        ),
    ],
)
def test_repeat(
    value: Any,
    n: int,
    dtype: pl.PolarsDataType,
    expected_dtype: pl.PolarsDataType,
) -> None:
    expected = pl.Series("repeat", [value] * n, dtype=expected_dtype)

    result_eager = pl.repeat(value, n=n, dtype=dtype, eager=True)
    assert_series_equal(result_eager, expected)

    result_lazy = pl.select(pl.repeat(value, n=n, dtype=dtype, eager=False)).to_series()
    assert_series_equal(result_lazy, expected)


def test_repeat_expr_input_eager() -> None:
    with pytest.raises(TypeError):
        pl.repeat(1, n=pl.lit(3), eager=True)


def test_repeat_expr_input_lazy() -> None:
    result = pl.select(pl.repeat(1, n=pl.lit(3))).to_series()
    expected = pl.Series("repeat", [1, 1, 1], dtype=pl.Int32)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("value", "n", "dtype", "expected_dtype"),
    [
        (1.0, 3, None, pl.Float64),
        (1, 2, pl.UInt8, pl.UInt8),
        (1, 0, pl.Int32, pl.Int32),
    ],
)
def test_ones(
    value: Any,
    n: int,
    dtype: pl.PolarsDataType,
    expected_dtype: pl.PolarsDataType,
) -> None:
    expected = pl.Series("ones", [value] * n, dtype=expected_dtype)

    result_eager = pl.ones(n=n, dtype=dtype, eager=True)
    assert_series_equal(result_eager, expected)

    result_lazy = pl.select(pl.ones(n=n, dtype=dtype, eager=False)).to_series()
    assert_series_equal(result_lazy, expected)


@pytest.mark.parametrize(
    ("value", "n", "dtype", "expected_dtype"),
    [
        (0.0, 3, None, pl.Float64),
        (0, 2, pl.UInt8, pl.UInt8),
        (0, 0, pl.Int32, pl.Int32),
    ],
)
def test_zeros(
    value: Any,
    n: int,
    dtype: pl.PolarsDataType,
    expected_dtype: pl.PolarsDataType,
) -> None:
    expected = pl.Series("zeros", [value] * n, dtype=expected_dtype)

    result_eager = pl.zeros(n=n, dtype=dtype, eager=True)
    assert_series_equal(result_eager, expected)

    result_lazy = pl.select(pl.zeros(n=n, dtype=dtype, eager=False)).to_series()
    assert_series_equal(result_lazy, expected)
