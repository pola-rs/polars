from __future__ import annotations

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series


@given(s=series(allow_null=True))
def test_drop_nulls_parametric(s: pl.Series) -> None:
    result = s.drop_nulls()
    assert result.len() == s.len() - s.null_count()

    filter_result = s.filter(s.is_not_null())
    assert_series_equal(result, filter_result)


def test_df_drop_nulls_struct() -> None:
    df = pl.DataFrame(
        {"x": [{"a": 1, "b": 2}, {"a": 1, "b": None}, {"a": None, "b": 2}, None]}
    )

    result = df.drop_nulls()

    expected = df.head(3)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("maintain_order", [False, True])
def test_drop_nulls_in_agg_25349(maintain_order: bool) -> None:
    lf = pl.LazyFrame({"a": [1, 2], "b": [None, 1]})
    q = lf.group_by("a", maintain_order=maintain_order).agg(
        pl.col.b.first().drop_nulls()
    )
    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": [1, 2], "b": [[], [1]]}),
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize("maintain_order", [False, True])
def test_drop_nulls_on_literal_25355(maintain_order: bool) -> None:
    df = pl.DataFrame({"key": [0, 1]})
    result = df.group_by("key", maintain_order=maintain_order).agg(
        x=pl.lit(0, dtype=pl.Int64).drop_nulls()
    )
    assert_frame_equal(
        result,
        df.with_columns(x=pl.lit([0], dtype=pl.List(pl.Int64))),
        check_row_order=maintain_order,
    )
