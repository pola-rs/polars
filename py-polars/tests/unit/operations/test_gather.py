import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal


def test_negative_index() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
    assert df.select(pl.col("a").gather([0, -1])).to_dict(as_series=False) == {
        "a": [1, 6]
    }
    assert df.group_by(pl.col("a") % 2).agg(b=pl.col("a").gather([0, -1])).sort(
        "a"
    ).to_dict(as_series=False) == {"a": [0, 1], "b": [[2, 6], [1, 5]]}


def test_gather_agg_schema() -> None:
    df = pl.DataFrame(
        {
            "group": [
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
            ],
            "value": [1, 98, 2, 3, 99, 4],
        }
    )
    assert (
        df.lazy()
        .group_by("group", maintain_order=True)
        .agg(pl.col("value").get(1))
        .collect_schema()["value"]
        == pl.Int64
    )


def test_gather_lit_single_16535() -> None:
    df = pl.DataFrame({"x": [1, 2, 2, 1], "y": [1, 2, 3, 4]})

    assert df.group_by(["x"], maintain_order=True).agg(pl.all().gather([1])).to_dict(
        as_series=False
    ) == {"x": [1, 2], "y": [[4], [3]]}


def test_list_get_null_offset_17248() -> None:
    df = pl.DataFrame({"material": [["PB", "PVC", "CI"], ["CI"], ["CI"]]})

    assert df.select(
        result=pl.when(pl.col.material.list.len() == 1).then("material").list.get(0),
    )["result"].to_list() == [None, "CI", "CI"]


def test_list_get_null_oob_17252() -> None:
    df = pl.DataFrame(
        {
            "name": ["BOB-3", "BOB", None],
        }
    )

    split = df.with_columns(pl.col("name").str.split("-"))
    assert split.with_columns(pl.col("name").list.get(0))["name"].to_list() == [
        "BOB",
        "BOB",
        None,
    ]


def test_list_get_null_on_oob_false_success() -> None:
    # test Series (single offset) with nulls
    expected = pl.Series("a", [2, None, 2], dtype=pl.Int64)
    s_nulls = pl.Series("a", [[1, 2], None, [1, 2, 3]])
    out = s_nulls.list.get(1, null_on_oob=False)
    assert_series_equal(out, expected)

    # test Expr (multiple offsets) with nulls
    df = s_nulls.to_frame().with_columns(pl.lit(1).alias("idx"))
    out = df.select(pl.col("a").list.get("idx", null_on_oob=True)).to_series()
    assert_series_equal(out, expected)

    # test Series (single offset) with no nulls
    expected = pl.Series("a", [2, 2, 2], dtype=pl.Int64)
    s_no_nulls = pl.Series("a", [[1, 2], [1, 2], [1, 2, 3]])
    out = s_no_nulls.list.get(1, null_on_oob=False)
    assert_series_equal(out, expected)

    # test Expr (multiple offsets) with no nulls
    df = s_no_nulls.to_frame().with_columns(pl.lit(1).alias("idx"))
    out = df.select(pl.col("a").list.get("idx", null_on_oob=True)).to_series()
    assert_series_equal(out, expected)


def test_list_get_null_on_oob_false_failure() -> None:
    # test Series (single offset) with nulls
    s_nulls = pl.Series("a", [[1, 2], None, [1, 2, 3]])
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        s_nulls.list.get(2, null_on_oob=False)

    # test Expr (multiple offsets) with nulls
    df = s_nulls.to_frame().with_columns(pl.lit(2).alias("idx"))
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        df.select(pl.col("a").list.get("idx", null_on_oob=False))

    # test Series (single offset) with no nulls
    s_no_nulls = pl.Series("a", [[1, 2], [1], [1, 2, 3]])
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        s_no_nulls.list.get(2, null_on_oob=False)

    # test Expr (multiple offsets) with no nulls
    df = s_no_nulls.to_frame().with_columns(pl.lit(2).alias("idx"))
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        df.select(pl.col("a").list.get("idx", null_on_oob=False))


def test_list_get_null_on_oob_true() -> None:
    # test Series (single offset) with nulls
    s_nulls = pl.Series("a", [[1, 2], None, [1, 2, 3]])
    out = s_nulls.list.get(2, null_on_oob=True)
    expected = pl.Series("a", [None, None, 3], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # test Expr (multiple offsets) with nulls
    df = s_nulls.to_frame().with_columns(pl.lit(2).alias("idx"))
    out = df.select(pl.col("a").list.get("idx", null_on_oob=True)).to_series()
    assert_series_equal(out, expected)

    # test Series (single offset) with no nulls
    s_no_nulls = pl.Series("a", [[1, 2], [1], [1, 2, 3]])
    out = s_no_nulls.list.get(2, null_on_oob=True)
    expected = pl.Series("a", [None, None, 3], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # test Expr (multiple offsets) with no nulls
    df = s_no_nulls.to_frame().with_columns(pl.lit(2).alias("idx"))
    out = df.select(pl.col("a").list.get("idx", null_on_oob=True)).to_series()
    assert_series_equal(out, expected)


def test_chunked_gather_phys_repr_17446() -> None:
    dfa = pl.DataFrame({"replace_unique_id": range(2)})

    for dt in [pl.Date, pl.Time, pl.Duration]:
        dfb = dfa.clone()
        dfb = dfb.with_columns(ds_start_date_right=pl.lit(None).cast(dt))
        dfb = pl.concat([dfb, dfb])

        assert dfa.join(dfb, how="left", on=pl.col("replace_unique_id")).shape == (4, 2)


def test_gather_str_col_18099() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "idx": [0, 0, 1]})
    assert df.with_columns(pl.col("foo").gather("idx")).to_dict(as_series=False) == {
        "foo": [1, 1, 2],
        "idx": [0, 0, 1],
    }
