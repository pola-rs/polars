from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import DuplicateError
from polars.testing import assert_frame_equal


def test_value_counts() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    result = s.value_counts()
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "count": [1, 2, 1]},
        schema_overrides={"count": pl.get_index_type()},
    )
    result_sorted = result.sort("a")
    assert_frame_equal(result_sorted, expected)

    out = pl.Series("a", [12, 3345, 12, 3, 4, 4, 1, 12]).value_counts(
        normalize=True, sort=True
    )
    assert out["proportion"].sum() == 1.0
    assert out.to_dict(as_series=False) == {
        "a": [12, 4, 3345, 3, 1],
        "proportion": [0.375, 0.25, 0.125, 0.125, 0.125],
    }


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_value_counts_streaming_group_by(sort: bool, normalize: bool) -> None:
    i = pl.int_range(0, 300_000)
    lf = pl.LazyFrame().select(
        a=pl.when(i % 101 == 0).then(None).otherwise(i * 7919 % 5003)
    )
    q = lf.select(pl.col("a").value_counts(sort=sort, normalize=normalize))

    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "group-by" in dot
    assert "columnar-function" not in dot

    result = q.collect().unnest("a")
    count_name = "proportion" if normalize else "count"
    if sort:
        assert result[count_name].is_sorted(descending=True)
    expected = lf.collect().to_series().value_counts(normalize=normalize)
    assert_frame_equal(result, expected, check_row_order=False)


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
    s = pl.Series("count", [1, 0, 1])

    # default name is 'count' ...
    with pytest.raises(
        DuplicateError,
        match="duplicate column names; change `name` to fix",
    ):
        s.value_counts()

    # ... but can customize that
    result = s.value_counts(name="n", sort=True)
    expected = pl.DataFrame(
        {"count": [1, 0], "n": [2, 1]}, schema_overrides={"n": pl.get_index_type()}
    )
    assert_frame_equal(result, expected)

    df = pl.DataFrame({"a": [None, 1, None, 2, 3]})
    result = df.select(pl.col("a").count())
    assert result.item() == 3

    result = df.group_by(1).agg(pl.col("a").count())
    assert result.to_dict(as_series=False) == {"literal": [1], "a": [3]}


def test_count() -> None:
    assert pl.Series([None, 1, None, 2, 3]).count() == 3
