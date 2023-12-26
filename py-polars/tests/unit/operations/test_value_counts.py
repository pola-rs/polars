from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_value_counts() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    result = s.value_counts()
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "count": [1, 2, 1]}, schema_overrides={"count": pl.UInt32}
    )
    result_sorted = result.sort("a")
    assert_frame_equal(result_sorted, expected)


def test_value_counts_logical_type() -> None:
    # test logical type
    df = pl.DataFrame({"a": ["b", "c"]}).with_columns(
        pl.col("a").cast(pl.Categorical).alias("ac")
    )
    out = df.select(pl.all().value_counts())
    assert out["ac"].struct.field("ac").dtype == pl.Categorical
    assert out["a"].struct.field("a").dtype == pl.String


def test_value_counts_expr() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d"],
        }
    )
    out = df.select(pl.col("id").value_counts(sort=True)).to_series().to_list()
    assert out == [
        {"id": "c", "count": 3},
        {"id": "b", "count": 2},
        {"id": "d", "count": 2},
        {"id": "a", "count": 1},
    ]

    # nested value counts. Then the series needs the name
    # 6200

    df = pl.DataFrame({"session": [1, 1, 1], "id": [2, 2, 3]})

    assert df.group_by("session").agg(
        pl.col("id").value_counts(sort=True).first()
    ).to_dict(as_series=False) == {"session": [1], "id": [{"id": 2, "count": 2}]}


def test_value_counts_duplicate_name() -> None:
    s = pl.Series("count", [1])

    with pytest.raises(pl.DuplicateError, match="count"):
        s.value_counts()

    def test_count() -> None:
        assert pl.Series([None, 1, None, 2, 3]).count() == 3

    df = pl.DataFrame({"a": [None, 1, None, 2, 3]})
    assert df.select(pl.col("a").count()).item() == 3

    assert df.group_by(1).agg(pl.col("a").count()).to_dict(as_series=False) == {
        "literal": [1],
        "a": [3],
    }
