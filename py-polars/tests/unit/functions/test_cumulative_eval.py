import pytest

import polars as pl
from polars.testing import assert_frame_equal


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
