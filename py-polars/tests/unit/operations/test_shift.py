from __future__ import annotations

from datetime import date

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_shift() -> None:
    a = pl.Series("a", [1, 2, 3])
    assert_series_equal(a.shift(1), pl.Series("a", [None, 1, 2]))
    assert_series_equal(a.shift(-1), pl.Series("a", [2, 3, None]))
    assert_series_equal(a.shift(-2), pl.Series("a", [3, None, None]))
    assert_series_equal(a.shift(-1, fill_value=10), pl.Series("a", [2, 3, 10]))


def test_shift_frame(fruits_cars: pl.DataFrame) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    out = df.select(pl.col("a").shift(1))
    assert_series_equal(out["a"], pl.Series("a", [None, 1, 2, 3, 4]))

    res = fruits_cars.lazy().shift(2).collect()

    expected = pl.DataFrame(
        {
            "A": [None, None, 1, 2, 3],
            "fruits": [None, None, "banana", "banana", "apple"],
            "B": [None, None, 5, 4, 3],
            "cars": [None, None, "beetle", "audi", "beetle"],
        }
    )
    assert_frame_equal(res, expected)

    # negative value
    res = fruits_cars.lazy().shift(-2).collect()
    for rows in [3, 4]:
        for cols in range(4):
            assert res[rows, cols] is None


def test_shift_and_fill() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = ldf.with_columns(
        pl.col("a").shift(n=-2, fill_value=pl.col("b").mean())
    ).collect()
    assert out["a"].null_count() == 0

    # use df method
    out = ldf.shift(n=2, fill_value=pl.col("b").std()).collect()
    assert out["a"].null_count() == 0


def test_shift_categorical() -> None:
    df = pl.Series("a", ["a", "b"], dtype=pl.Categorical).to_frame()

    s = df.with_columns(pl.col("a").shift(fill_value="c"))["a"]
    assert s.dtype == pl.Categorical
    assert s.to_list() == ["c", "a"]


def test_shift_frame_with_fill() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    result = df.shift(fill_value=0)
    expected = pl.DataFrame(
        {
            "foo": [0, 1, 2],
            "bar": [0, 6, 7],
            "ham": ["0", "a", "b"],
        }
    )
    assert_frame_equal(result, expected)


def test_shift_and_fill_group_logicals() -> None:
    df = pl.DataFrame(
        [
            (date(2001, 1, 2), "A"),
            (date(2001, 1, 3), "A"),
            (date(2001, 1, 4), "A"),
            (date(2001, 1, 3), "B"),
            (date(2001, 1, 4), "B"),
        ],
        schema=["d", "s"],
    )
    result = df.select(pl.col("d").shift(fill_value=pl.col("d").max(), n=-1).over("s"))

    assert result.dtypes == [pl.Date]


def test_shift_and_fill_deprecated() -> None:
    a = pl.Series("a", [1, 2, 3])

    with pytest.deprecated_call():
        result = a.shift_and_fill(100, n=-1)

    expected = pl.Series("a", [2, 3, 100])
    assert_series_equal(result, expected)


def test_shift_and_fill_frame_deprecated() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.deprecated_call():
        result = lf.shift_and_fill(100, n=1)

    expected = pl.LazyFrame({"a": [100, 1, 2], "b": [100, 4, 5]})
    assert_frame_equal(result, expected)
