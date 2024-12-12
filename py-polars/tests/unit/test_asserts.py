import pytest

import polars as pl


def test_assert() -> None:
    df = pl.DataFrame(
        {
            "a": [2, 4, 6, 8, 10],
            "b": ["abc", "wow", "some", "strange", "text"],
        }
    )

    df.assert_err(pl.col.a <= 10)
    with pytest.raises(pl.exceptions.AssertionFailedError):
        df.assert_err(pl.col.a < 8)


def test_assert_pred_pd() -> None:
    lf = pl.LazyFrame(
        {
            "a": [2, 4, 6, 8, 10],
            "b": ["abc", "wow", "some", "strange", "text"],
        }
    )
    with pytest.raises(pl.exceptions.AssertionFailedError):
        lf.assert_err(pl.col.a < 8).collect()
    lf.assert_err(pl.col.a < 8).filter(pl.col.a < 8).collect()
    with pytest.raises(pl.exceptions.AssertionFailedError):
        lf.assert_err(pl.col.a < 8, allow_predicate_pushdown=False).filter(
            pl.col.a < 8
        ).collect()
    lf.filter(pl.col.a < 8).assert_err(
        pl.col.a < 8, allow_predicate_pushdown=False
    ).collect()
