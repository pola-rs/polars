from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars import StringCache
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from typing import Any

    from polars.type_aliases import PolarsDataType


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


def test_shift_expr() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = ldf.select(pl.col("a").shift(n=pl.col("b").min())).collect()
    assert out.to_dict(as_series=False) == {"a": [None, 1, 2, 3, 4]}

    out = ldf.select(
        pl.col("a").shift(pl.col("b").min(), fill_value=pl.col("b").max())
    ).collect()
    assert out.to_dict(as_series=False) == {"a": [5, 1, 2, 3, 4]}

    # use df method
    out = ldf.shift(pl.lit(3)).collect()
    assert out.to_dict(as_series=False) == {
        "a": [None, None, None, 1, 2],
        "b": [None, None, None, 1, 2],
    }
    out = ldf.shift(pl.lit(2), fill_value=pl.col("b").max()).collect()
    assert out.to_dict(as_series=False) == {"a": [5, 5, 1, 2, 3], "b": [5, 5, 1, 2, 3]}


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


@pytest.mark.parametrize(
    ("shift_amt", "idx_out"),
    [
        (0, [0, 1, 2]),
        (1, [2, 0, 1]),
        (2, [1, 2, 0]),
        (3, [0, 1, 2]),
        (-1, [1, 2, 0]),
        (-2, [2, 0, 1]),
        (-3, [0, 1, 2]),
    ],
)
@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([1, 2, 3], pl.Int32),
        (["a", "b", "c"], pl.String),
        (["a", "b", "c"], pl.Categorical),
        ([[1], [2, 2], [3, 3, 3]], pl.List),
        ([[1, 1], [2, 2], [3, 3]], pl.Array(pl.Int32, 2)),
        ([b"0", b"1", b"2"], pl.Binary),
        ([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)], pl.Date),
        (
            [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            pl.Datetime,
        ),
    ],
)
@StringCache()
def test_circshift(
    values: list[Any], dtype: PolarsDataType, shift_amt: int, idx_out: list[int]
) -> None:
    a = pl.Series("a", values, dtype=dtype)
    values_out = [values[i] for i in idx_out]
    expected = pl.Series("a", values_out, dtype=dtype)
    assert_series_equal(a.circshift(shift_amt), expected)


def test_circshift_frame(fruits_cars: pl.DataFrame) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    out = df.select(pl.col("a").circshift(1))
    assert_series_equal(out["a"], pl.Series("a", [5, 1, 2, 3, 4]))

    res = fruits_cars.lazy().circshift(2).collect()
    expected = pl.DataFrame(
        {
            "A": [4, 5, 1, 2, 3],
            "fruits": ["apple", "banana", "banana", "banana", "apple"],
            "B": [2, 1, 5, 4, 3],
            "cars": ["beetle", "beetle", "beetle", "audi", "beetle"],
        }
    )
    assert_frame_equal(res, expected)

    # negative value
    res = fruits_cars.lazy().circshift(-2).collect()
    expected = pl.DataFrame(
        {
            "A": [3, 4, 5, 1, 2],
            "fruits": ["apple", "apple", "banana", "banana", "banana"],
            "B": [3, 2, 1, 5, 4],
            "cars": ["beetle", "beetle", "beetle", "beetle", "audi"],
        }
    )
    assert_frame_equal(res, expected)


def test_circshift_expr() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = ldf.select(pl.col("a").circshift(n=pl.col("b").min())).collect()
    assert out.to_dict(as_series=False) == {"a": [5, 1, 2, 3, 4]}

    out = ldf.select(pl.col("a").circshift(pl.col("b").min())).collect()
    assert out.to_dict(as_series=False) == {"a": [5, 1, 2, 3, 4]}

    # use df method
    out = ldf.circshift(pl.lit(3)).collect()
    assert out.to_dict(as_series=False) == {
        "a": [3, 4, 5, 1, 2],
        "b": [3, 4, 5, 1, 2],
    }
    out = ldf.circshift(pl.lit(2)).collect()
    assert out.to_dict(as_series=False) == {"a": [4, 5, 1, 2, 3], "b": [4, 5, 1, 2, 3]}


def test_circshift_categorical() -> None:
    df = pl.Series("a", ["a", "b"], dtype=pl.Categorical).to_frame()

    s = df.with_columns(pl.col("a").circshift())["a"]
    assert s.dtype == pl.Categorical
    assert s.to_list() == ["b", "a"]
