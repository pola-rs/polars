import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

left = pl.DataFrame({"a": [42, 13, 37], "b": [3, 8, 9]})
right = pl.DataFrame({"a": [5, 10, 1996], "b": [1, 5, 7]})
expected = pl.DataFrame(
    {
        "a": [5, 42, 10, 1996, 13, 37],
        "b": [1, 3, 5, 7, 8, 9],
    }
)

lf = left.lazy().merge_sorted(right.lazy(), "b")


@pytest.mark.parametrize("streaming", [False, True])
def test_merge_sorted(streaming: bool) -> None:
    assert_frame_equal(
        lf.collect(new_streaming=streaming),  # type: ignore[call-overload]
        expected,
    )


def test_merge_sorted_pred_pd() -> None:
    assert_frame_equal(
        lf.filter(pl.col.b > 30).collect(),
        expected.filter(pl.col.b > 30),
    )
    assert_frame_equal(
        lf.filter(pl.col.a < 6).collect(),
        expected.filter(pl.col.a < 6),
    )


def test_merge_sorted_proj_pd() -> None:
    assert_frame_equal(
        lf.select("b").collect(),
        lf.collect().select("b"),
    )
    assert_frame_equal(
        lf.select("a").collect(),
        lf.collect().select("a"),
    )


@pytest.mark.parametrize("precision", [2, 3])
def test_merge_sorted_decimal_20990(precision: int) -> None:
    dtype = pl.Decimal(precision=precision, scale=1)
    s = pl.Series("a", ["1.0", "0.1"], dtype)
    df = pl.DataFrame([s.sort()])
    result = df.lazy().merge_sorted(df.lazy(), "a").collect().get_column("a")
    expected = pl.Series("a", ["0.1", "0.1", "1.0", "1.0"], dtype)
    assert_series_equal(result, expected)


def test_merge_sorted_categorical() -> None:
    left = pl.Series("a", ["a", "b"], pl.Categorical()).sort().to_frame()
    right = pl.Series("a", ["a", "b", "b"], pl.Categorical()).sort().to_frame()
    result = left.merge_sorted(right, "a").get_column("a")
    expected = pl.Series("a", ["a", "a", "b", "b", "b"], pl.Categorical())
    assert_series_equal(result, expected)

    right = pl.Series("a", ["b", "a"], pl.Categorical()).sort().to_frame()
    with pytest.raises(
        ComputeError, match="can only merge-sort categoricals with the same categories"
    ):
        left.merge_sorted(right, "a")
