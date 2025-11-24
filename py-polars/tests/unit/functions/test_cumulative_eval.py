import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_cumulative_eval_sum() -> None:
    assert_frame_equal(
        pl.DataFrame({"a": [1, 2, 3]}).with_columns(
            b=pl.col.a.cumulative_eval(pl.element().sum())
        ),
        pl.DataFrame({"a": [1, 2, 3], "b": [1, 3, 6]}),
    )


def test_cumulative_eval_group_by() -> None:
    assert_frame_equal(
        pl.DataFrame({"a": [1, 2, 3, 2, 3, 4, 3], "g": [1, 1, 1, 2, 2, 2, 3]})
        .group_by("g")
        .agg(b=pl.col.a.cumulative_eval(pl.element().sum())),
        pl.DataFrame({"g": [1, 2, 3], "b": [[1, 3, 6], [2, 5, 9], [3]]}),
        check_row_order=False,
    )


def test_cumulative_eval_deny_non_scalar() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="scalar"):
        (
            pl.DataFrame({"a": [1, 2, 3]}).with_columns(
                b=pl.col.a.cumulative_eval(pl.element() + 1)
            ),
        )


def test_cumulative_eval_empty() -> None:
    s = pl.Series("a", [], pl.Int64)
    assert_series_equal(s.cumulative_eval(pl.element().first()), s)


def test_cumulative_eval_samples() -> None:
    assert_series_equal(
        pl.Series("a", [None, None, 1, 2, 3, None, None], pl.Int64).cumulative_eval(
            pl.element().first(), min_samples=3
        ),
        pl.Series("a", [None, None, None, None, None, None, None], pl.Int64),
    )

    assert_series_equal(
        pl.Series("a", [None, None, 1, 2, 3, None, None], pl.Int64).cumulative_eval(
            pl.element().min(), min_samples=3
        ),
        pl.Series("a", [None, None, None, None, 1, 1, 1], pl.Int64),
    )


def test_cumulative_eval_length_preserving_streaming_25293() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    q = df.lazy().with_columns(
        pl.col("a")
        .first()
        .cumulative_eval(
            pl.element().map_batches(lambda x: x.max(), pl.Int64, returns_scalar=True)
        )
    )
    expected = pl.DataFrame({"a": [1, 1, 1]})
    assert_frame_equal(q.collect(engine="in-memory"), expected)
    assert_frame_equal(q.collect(engine="streaming"), expected)
