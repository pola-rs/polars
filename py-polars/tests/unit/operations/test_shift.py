from __future__ import annotations

from datetime import date

import pytest

import polars as pl
from polars.exceptions import ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_shift() -> None:
    a = pl.Series("a", [1, 2, 3])
    assert_series_equal(a.shift(1), pl.Series("a", [None, 1, 2]))
    assert_series_equal(a.shift(-1), pl.Series("a", [2, 3, None]))
    assert_series_equal(a.shift(-2), pl.Series("a", [3, None, None]))
    assert_series_equal(a.shift(-1, fill_value=10), pl.Series("a", [2, 3, 10]))


def test_shift_object() -> None:
    a = pl.Series("a", [1, 2, 3], dtype=pl.Object)
    assert a.shift(1).to_list() == [None, 1, 2]
    assert a.shift(-1).to_list() == [2, 3, None]
    assert a.shift(-2, fill_value=pl.lit(0, dtype=pl.Object)).to_list() == [3, 0, 0]
    assert a.shift(1).dtype == pl.Object


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


def test_shift_fill_value() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = ldf.with_columns(
        pl.col("a").shift(n=-2, fill_value=pl.col("b").mean())
    ).collect()
    assert not out["a"].has_nulls()

    # use df method
    out = ldf.shift(n=2, fill_value=pl.col("b").std()).collect()
    assert not out["a"].has_nulls()


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


def test_shift_fill_value_group_logicals() -> None:
    df = pl.DataFrame(
        [
            (date(2001, 1, 2), "A"),
            (date(2001, 1, 3), "A"),
            (date(2001, 1, 4), "A"),
            (date(2001, 1, 3), "B"),
            (date(2001, 1, 4), "B"),
        ],
        schema=["d", "s"],
        orient="row",
    )
    result = df.select(pl.col("d").shift(fill_value=pl.col("d").max(), n=-1).over("s"))

    assert result.dtypes == [pl.Date]


def test_shift_n_null() -> None:
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32)})
    out = df.shift(None)  # type: ignore[arg-type]
    expected = pl.DataFrame({"a": pl.Series([None, None, None], dtype=pl.Int32)})
    assert_frame_equal(out, expected)

    out = df.shift(None, fill_value=1)  # type: ignore[arg-type]
    assert_frame_equal(out, expected)

    out = df.select(pl.col("a").shift(None))  # type: ignore[arg-type]
    assert_frame_equal(out, expected)

    out = df.select(pl.col("a").shift(None, fill_value=1))  # type: ignore[arg-type]
    assert_frame_equal(out, expected)


def test_shift_n_nonscalar() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    with pytest.raises(
        ShapeError,
        match="'n' must be a scalar value",
    ):
        # Note: Expressions are not in the signature for `n`, but they work.
        # We can still verify that n is scalar up-front.
        df.shift(pl.col("b"), fill_value=1)  # type: ignore[arg-type]

    with pytest.raises(
        ShapeError,
        match="'n' must be a scalar value",
    ):
        df.select(pl.col("a").shift(pl.col("b"), fill_value=1))


def test_shift_fill_value_nonscalar() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    with pytest.raises(
        ShapeError,
        match="'fill_value' must be a scalar value",
    ):
        df.shift(1, fill_value=pl.col("b"))

    with pytest.raises(
        ShapeError,
        match="'fill_value' must be a scalar value",
    ):
        df.select(pl.col("a").shift(1, fill_value=pl.col("b")))


def test_shift_array_list_eval_24672() -> None:
    s = pl.Series([[[1], [2], [3]]], dtype=pl.List(pl.Array(pl.Int64, 1)))
    expected = pl.Series([[None, [1], [2]]], dtype=pl.List(pl.Array(pl.Int64, 1)))
    out = s.list.eval(pl.element().shift())
    assert_series_equal(expected, out)


def test_streaming_shift_25226() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]})

    q = df.lazy().with_columns(b=pl.col("a").shift(), c=pl.col("a").min())
    assert_frame_equal(
        q.collect(),
        df.with_columns(b=pl.Series([None, 1, 2, 3]), c=pl.lit(1, pl.Int64)),
    )

    q = df.lazy().with_columns(b=pl.col("a").shift(n=-1), c=pl.col("a").min())
    assert_frame_equal(
        q.collect(),
        df.with_columns(b=pl.Series([2, 3, 4, None]), c=pl.lit(1, pl.Int64)),
    )


def test_streaming_shift_with_head_26098() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    q = df.lazy().select(pl.col("a").shift(-1)).head(1)
    assert_frame_equal(
        q.collect(engine="streaming"),
        pl.DataFrame({"a": [2]}),
    )
