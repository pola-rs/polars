from datetime import date, datetime
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, OutOfBoundsError
from polars.testing import assert_series_equal


@pytest.mark.parametrize(
    "input",
    [
        (),
        [],
        pl.Series(),
        pl.Series(dtype=pl.Int8),
        np.array([]),
    ],
)
def test_scatter_noop(input: Any) -> None:
    s = pl.Series("s", [1, 2, 3])
    s.scatter(input, 8)
    assert s.to_list() == [1, 2, 3]


def test_scatter() -> None:
    s = pl.Series("s", [1, 2, 3])

    # set new values, one index at a time
    s.scatter(0, 8)
    s.scatter([1], None)
    assert s.to_list() == [8, None, 3]

    # set new value at multiple indexes in one go
    s.scatter([0, 2], None)
    assert s.to_list() == [None, None, None]

    # try with different series dtype
    s = pl.Series("s", ["a", "b", "c"])
    s.scatter((1, 2), "x")
    assert s.to_list() == ["a", "x", "x"]
    assert s.scatter([0, 2], 0.12345).to_list() == ["0.12345", "x", "0.12345"]

    # set multiple values values
    s = pl.Series(["z", "z", "z"])
    assert s.scatter([0, 1], ["a", "b"]).to_list() == ["a", "b", "z"]
    s = pl.Series([True, False, True])
    assert s.scatter([0, 1], [False, True]).to_list() == [False, True, True]

    # set negative indices
    a = pl.Series("r", range(5))
    a[-2] = None
    a[-5] = None
    assert a.to_list() == [None, 1, 2, None, 4]

    a = pl.Series("x", [1, 2])
    with pytest.raises(OutOfBoundsError):
        a[-100] = None
    assert_series_equal(a, pl.Series("x", [1, 2]))


def test_index_with_None_errors_16905() -> None:
    s = pl.Series("s", [1, 2, 3])
    with pytest.raises(ComputeError, match="index values should not be null"):
        s[[1, None]] = 5
    # The error doesn't trash the series, as it used to:
    assert_series_equal(s, pl.Series("s", [1, 2, 3]))


def test_object_dtype_16905() -> None:
    obj = object()
    s = pl.Series("s", [obj, 27], dtype=pl.Object)
    # This operation is not semantically wrong, it might be supported in the
    # future, but for now it isn't.
    with pytest.raises(InvalidOperationError):
        s[0] = 5
    # The error doesn't trash the series, as it used to:
    assert s.dtype == pl.Object
    assert s.name == "s"
    assert s.to_list() == [obj, 27]


def test_scatter_datetime() -> None:
    s = pl.Series("dt", [None, datetime(2024, 1, 31)])
    result = s.scatter(0, datetime(2022, 2, 2))
    expected = pl.Series("dt", [datetime(2022, 2, 2), datetime(2024, 1, 31)])
    assert_series_equal(result, expected)


def test_scatter_logical_all_null() -> None:
    s = pl.Series("dt", [None, None], dtype=pl.Date)
    result = s.scatter(0, date(2022, 2, 2))
    expected = pl.Series("dt", [date(2022, 2, 2), None])
    assert_series_equal(result, expected)
