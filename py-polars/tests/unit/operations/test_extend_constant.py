from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    ("const", "dtype"),
    [
        (1, pl.Int8),
        (4, pl.UInt32),
        (4.5, pl.Float32),
        (None, pl.Float64),
        ("白鵬翔", pl.String),
        (date.today(), pl.Date),
        (datetime.now(), pl.Datetime("ns")),
        (time(23, 59, 59), pl.Time),
        (timedelta(hours=7, seconds=123), pl.Duration("ms")),
    ],
)
def test_extend_constant(const: Any, dtype: PolarsDataType) -> None:
    df = pl.DataFrame({"a": pl.Series("s", [None], dtype=dtype)})

    expected_df = pl.DataFrame(
        {"a": pl.Series("s", [None, const, const, const], dtype=dtype)}
    )

    assert_frame_equal(df.select(pl.col("a").extend_constant(const, 3)), expected_df)

    s = pl.Series("s", [None], dtype=dtype)
    expected = pl.Series("s", [None, const, const, const], dtype=dtype)
    assert_series_equal(s.extend_constant(const, 3), expected)

    # test n expr
    expected = pl.Series("s", [None, const, const], dtype=dtype)
    assert_series_equal(s.extend_constant(const, pl.Series([2])), expected)

    # test value expr
    expected = pl.Series("s", [None, const, const, const], dtype=dtype)
    assert_series_equal(s.extend_constant(pl.Series([const], dtype=dtype), 3), expected)


@pytest.mark.parametrize(
    ("const", "dtype"),
    [
        (1, pl.Int8),
        (4, pl.UInt32),
        (4.5, pl.Float32),
        (None, pl.Float64),
        ("白鵬翔", pl.String),
        (date.today(), pl.Date),
        (datetime.now(), pl.Datetime("ns")),
        (time(23, 59, 59), pl.Time),
        (timedelta(hours=7, seconds=123), pl.Duration("ms")),
    ],
)
def test_extend_constant_arr(const: Any, dtype: PolarsDataType) -> None:
    """
    Test extend_constant in pl.List array.

    NOTE: This function currently fails when the Series is a list with a single [None]
          value. Hence, this function does not begin with [[None]], but [[const]].
    """
    s = pl.Series("s", [[const]], dtype=pl.List(dtype))

    expected = pl.Series("s", [[const, const, const, const]], dtype=pl.List(dtype))

    assert_series_equal(s.list.eval(pl.element().extend_constant(const, 3)), expected)


def test_extend_by_not_uint_expr() -> None:
    s = pl.Series("s", [1])
    with pytest.raises(ComputeError, match="value and n should have unit length"):
        s.extend_constant(pl.Series([2, 3]), 3)
    with pytest.raises(ComputeError, match="value and n should have unit length"):
        s.extend_constant(2, pl.Series([3, 4]))
