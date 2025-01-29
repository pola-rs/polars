import pytest

import polars as pl
from polars.testing import assert_frame_equal

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
