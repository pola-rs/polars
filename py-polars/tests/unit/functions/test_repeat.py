from datetime import date, datetime, time, timedelta
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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
        (datetime(2023, 2, 2), 3, None, pl.Datetime),
        (date(2023, 2, 2), 3, None, pl.Date),
        (time(10, 15), 1, None, pl.Time),
        (timedelta(hours=3), 10, None, pl.Duration),
        (8, 2, pl.UInt8, pl.UInt8),
        (date(2023, 2, 2), 3, pl.Datetime, pl.Datetime),
        (7.5, 5, pl.UInt16, pl.UInt16),
    ],
)
def test_repeat(
    value: Any,
    n: int,
    dtype: pl.PolarsDataType,
    expected_dtype: pl.PolarsDataType,
) -> None:
    expected = pl.Series("repeat", [value] * n).cast(expected_dtype)

    result_eager = pl.repeat(value, n=n, dtype=dtype, eager=True)
    assert_series_equal(result_eager, expected)

    result_lazy = pl.select(pl.repeat(value, n=n, dtype=dtype, eager=False)).to_series()
    assert_series_equal(result_lazy, expected)


def test_repeat_expr_input_eager() -> None:
    result = pl.select(pl.repeat(1, n=pl.lit(3), eager=True)).to_series()
    expected = pl.Series("repeat", [1, 1, 1], dtype=pl.Int32)
    assert_series_equal(result, expected)


def test_repeat_expr_input_lazy() -> None:
    df = pl.DataFrame({"a": [3, 2, 1]})
    result = df.select(pl.repeat(1, n=pl.col("a"))).to_series()
    expected = pl.Series("repeat", [1, 1, 1], dtype=pl.Int32)
    assert_series_equal(result, expected)

    df = pl.DataFrame({"a": [3, 2, 1]})
    assert df.select(pl.repeat(pl.sum("a"), n=2)).to_series().to_list() == [6, 6]


def test_repeat_n_zero() -> None:
    assert pl.repeat(1, n=0, eager=True).len() == 0


@pytest.mark.parametrize(
    "n",
    [1.5, 2.0, date(1971, 1, 2), "hello"],
)
def test_repeat_n_non_integer(n: Any) -> None:
    with pytest.raises(pl.SchemaError, match="expected expression of dtype 'integer'"):
        pl.repeat(1, n=pl.lit(n), eager=True)


def test_repeat_n_empty() -> None:
    df = pl.DataFrame(schema={"a": pl.Int32})
    with pytest.raises(pl.OutOfBoundsError, match="index 0 is out of bounds"):
        df.select(pl.repeat(1, n=pl.col("a")))


def test_repeat_n_negative() -> None:
    with pytest.raises(pl.ComputeError, match="could not parse value '-1' as a size"):
        pl.repeat(1, n=-1, eager=True)


@pytest.mark.parametrize(
    ("n", "dtype", "expected_dtype"),
    [
        (3, None, pl.Float64),
        (2, pl.UInt8, pl.UInt8),
        (0, pl.Int32, pl.Int32),
    ],
)
def test_ones(
    n: int,
    dtype: pl.PolarsDataType,
    expected_dtype: pl.PolarsDataType,
) -> None:
    expected = pl.Series("ones", [1] * n, dtype=expected_dtype)

    result_eager = pl.ones(n=n, dtype=dtype, eager=True)
    assert_series_equal(result_eager, expected)

    result_lazy = pl.select(pl.ones(n=n, dtype=dtype, eager=False)).to_series()
    assert_series_equal(result_lazy, expected)


@pytest.mark.parametrize(
    ("n", "dtype", "expected_dtype"),
    [
        (3, None, pl.Float64),
        (2, pl.UInt8, pl.UInt8),
        (0, pl.Int32, pl.Int32),
    ],
)
def test_zeros(
    n: int,
    dtype: pl.PolarsDataType,
    expected_dtype: pl.PolarsDataType,
) -> None:
    expected = pl.Series("zeros", [0] * n, dtype=expected_dtype)

    result_eager = pl.zeros(n=n, dtype=dtype, eager=True)
    assert_series_equal(result_eager, expected)

    result_lazy = pl.select(pl.zeros(n=n, dtype=dtype, eager=False)).to_series()
    assert_series_equal(result_lazy, expected)


def test_repeat_by_logical_dtype() -> None:
    with pl.StringCache():
        df = pl.DataFrame(
            {
                "repeat": [1, 2, 3],
                "date": [date(2021, 1, 1)] * 3,
                "cat": ["a", "b", "c"],
            },
            schema={"repeat": pl.Int32, "date": pl.Date, "cat": pl.Categorical},
        )
        out = df.select(
            pl.col("date").repeat_by("repeat"), pl.col("cat").repeat_by("repeat")
        )

        expected_df = pl.DataFrame(
            {
                "date": [
                    [date(2021, 1, 1)],
                    [date(2021, 1, 1), date(2021, 1, 1)],
                    [date(2021, 1, 1), date(2021, 1, 1), date(2021, 1, 1)],
                ],
                "cat": [["a"], ["b", "b"], ["c", "c", "c"]],
            },
            schema={"date": pl.List(pl.Date), "cat": pl.List(pl.Categorical)},
        )

        assert_frame_equal(out, expected_df)
