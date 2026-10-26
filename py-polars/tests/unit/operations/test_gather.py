from __future__ import annotations

import io
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import ComputeError, OutOfBoundsError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import EngineType


def test_negative_index() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
    assert df.select(pl.col("a").gather([0, -1])).to_dict(as_series=False) == {
        "a": [1, 6]
    }
    assert_frame_equal(
        df.group_by(pl.col("a") % 2).agg(b=pl.col("a").gather([0, -1])),
        pl.DataFrame({"a": [0, 1], "b": [[2, 6], [1, 5]]}),
        check_row_order=False,
    )


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


def test_gather_list_19243() -> None:
    df = pl.DataFrame({"a": [[0.1, 0.2, 0.3]]})
    assert df.with_columns(pl.lit([0]).alias("c")).with_columns(
        gather=pl.col("a").list.gather(pl.col("c"), null_on_oob=True)
    ).to_dict(as_series=False) == {
        "a": [[0.1, 0.2, 0.3]],
        "c": [[0]],
        "gather": [[0.1]],
    }


def test_gather_array_list_null_19302() -> None:
    data = pl.DataFrame(
        {"data": [None]}, schema_overrides={"data": pl.List(pl.Array(pl.Float32, 1))}
    )
    assert data.select(pl.col("data").list.get(0)).to_dict(as_series=False) == {
        "data": [None]
    }


def test_gather_array() -> None:
    a = np.arange(16).reshape(-1, 2, 2)
    s = pl.Series(a)

    for idx in [[1, 2], [0, 0], [1, 0], [1, 1, 1, 1, 1, 1, 1, 1]]:
        assert (s.gather(idx).to_numpy() == a[idx]).all()

    v = s[[0, 1, None, 3]]  # type: ignore[list-item]
    assert v[2] is None


def test_gather_array_outer_validity_19482() -> None:
    s = (
        pl.Series([[1], [1]], dtype=pl.Array(pl.Int64, 1))
        .to_frame()
        .select(pl.when(pl.int_range(pl.len()) == 0).then(pl.first()))
        .to_series()
    )

    expect = pl.Series([[1], None], dtype=pl.Array(pl.Int64, 1))
    assert_series_equal(s, expect)
    assert_series_equal(s.gather([0, 1]), expect)


def test_gather_len_19561() -> None:
    N = 4
    df = pl.DataFrame({"foo": ["baz"] * N, "bar": range(N)})

    idxs = (
        pl.int_range(1, N).repeat_by(pl.int_range(1, N)).list.explode(keep_nulls=False)
    )
    gather = pl.col("bar").gather(idxs).alias("gather")

    assert df.group_by("foo").agg(gather.len()).to_dict(as_series=False) == {
        "foo": ["baz"],
        "gather": [6],
    }


def test_gather_agg_group_update_scalar() -> None:
    # If `gather` doesn't update groups properly, `first` will try to access
    # index 2 (the original index of the first element of group `1`), but gather
    # outputs only two elements (one for each group), leading to an out of
    # bounds access.
    df = (
        pl.DataFrame({"gid": [0, 0, 1, 1], "x": ["0:0", "0:1", "1:0", "1:1"]})
        .lazy()
        .group_by("gid", maintain_order=True)
        .agg(x_at_gid=pl.col("x").gather(pl.col("gid").last()).first())
        .collect(optimizations=pl.QueryOptFlags.none())
    )
    expected = pl.DataFrame({"gid": [0, 1], "x_at_gid": ["0:0", "1:1"]})
    assert_frame_equal(df, expected)


def test_gather_agg_group_update_literal() -> None:
    # If `gather` doesn't update groups properly, `first` will try to access
    # index 2 (the original index of the first element of group `1`), but gather
    # outputs only two elements (one for each group), leading to an out of
    # bounds access.
    df = (
        pl.DataFrame({"gid": [0, 0, 1], "x": ["0:0", "0:1", "1:0"]})
        .lazy()
        .group_by("gid", maintain_order=True)
        .agg(x_at_0=pl.col("x").gather(0).first())
        .collect(optimizations=pl.QueryOptFlags.none())
    )
    expected = pl.DataFrame({"gid": [0, 1], "x_at_0": ["0:0", "1:0"]})
    assert_frame_equal(df, expected)


def test_gather_agg_group_update_negative() -> None:
    # If `gather` doesn't update groups properly, `first` will try to access
    # index 2 (the original index of the first element of group `1`), but gather
    # outputs only two elements (one for each group), leading to an out of
    # bounds access.
    df = (
        pl.DataFrame({"gid": [0, 0, 1], "x": ["0:0", "0:1", "1:0"]})
        .lazy()
        .group_by("gid", maintain_order=True)
        .agg(x_last=pl.col("x").gather(-1).first())
        .collect(optimizations=pl.QueryOptFlags.none())
    )
    expected = pl.DataFrame({"gid": [0, 1], "x_last": ["0:1", "1:0"]})
    assert_frame_equal(df, expected)


def test_gather_agg_group_update_multiple() -> None:
    # If `gather` doesn't update groups properly, `first` will try to access
    # index 4 (the original index of the first element of group `1`), but gather
    # outputs only four elements (two for each group), leading to an out of
    # bounds access.
    df = (
        pl.DataFrame(
            {
                "gid": [0, 0, 0, 0, 1, 1],
                "x": ["0:0", "0:1", "0:2", "0:3", "1:0", "1:1"],
            }
        )
        .lazy()
        .group_by("gid", maintain_order=True)
        .agg(x_at_0=pl.col("x").gather([0, 1]).first())
        .collect(optimizations=pl.QueryOptFlags.none())
    )
    expected = pl.DataFrame({"gid": [0, 1], "x_at_0": ["0:0", "1:0"]})
    assert_frame_equal(df, expected)


def test_get_agg_group_update_literal_21610() -> None:
    df = (
        pl.DataFrame(
            {
                "group": [100, 100, 100, 200, 200, 200],
                "value": [1, 2, 3, 2, 3, 4],
            }
        )
        .group_by("group", maintain_order=True)
        .agg(pl.col("value") - pl.col("value").get(0))
    )

    expected = pl.DataFrame({"group": [100, 200], "value": [[0, 1, 2], [0, 1, 2]]})
    assert_frame_equal(df, expected)


def test_get_agg_group_update_scalar_21610() -> None:
    df = (
        pl.DataFrame(
            {
                "group": [100, 100, 100, 200, 200, 200],
                "value": [1, 2, 3, 2, 3, 4],
            }
        )
        .group_by("group", maintain_order=True)
        .agg(pl.col("value") - pl.col("value").get(pl.col("value").first()))
    )

    expected = pl.DataFrame({"group": [100, 200], "value": [[-1, 0, 1], [-2, -1, 0]]})
    assert_frame_equal(df, expected)


def test_get_dt_truncate_21533() -> None:
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2016, 1, 1),
                pl.datetime(2017, 12, 31),
                interval="1d",
                eager=True,
            ),
        }
    ).with_columns(
        month=pl.col.timestamp.dt.month(),
    )

    report = df.group_by("month", maintain_order=True).agg(
        trunc_ts=pl.col.timestamp.get(0).dt.truncate("1m")
    )
    assert report.shape == (12, 2)


@pytest.mark.parametrize("maintain_order", [False, True])
def test_gather_group_by_23696(maintain_order: bool) -> None:
    df = (
        pl.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [0, 0, 1, 1],
                "c": [0, 0, -1, -1],
            }
        )
        .group_by(pl.col.a % 2, maintain_order=maintain_order)
        .agg(
            get_first=pl.col.a.get(pl.col.b.get(0)),
            get_last=pl.col.a.get(pl.col.b.get(1)),
            normal=pl.col.a.gather(pl.col.b),
            signed=pl.col.a.gather(pl.col.c),
            drop_nulls=pl.col.a.gather(pl.col.b.drop_nulls()),
            drop_nulls_signed=pl.col.a.gather(pl.col.c.drop_nulls()),
            literal=pl.col.a.gather([0, 1]),
            literal_signed=pl.col.a.gather([0, -1]),
        )
    )

    expected = pl.DataFrame(
        {
            "a": [1, 0],
            "get_first": [1, 2],
            "get_last": [3, 4],
            "normal": [[1, 3], [2, 4]],
            "signed": [[1, 3], [2, 4]],
            "drop_nulls": [[1, 3], [2, 4]],
            "drop_nulls_signed": [[1, 3], [2, 4]],
            "literal": [[1, 3], [2, 4]],
            "literal_signed": [[1, 3], [2, 4]],
        }
    )

    assert_frame_equal(df, expected, check_row_order=maintain_order)


def test_gather_invalid_indices_groupby_24182() -> None:
    df = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.group_by(True).agg(pl.col("x").gather(pl.lit("y")))


@pytest.mark.parametrize("maintain_order", [False, True])
def test_gather_group_by_lit(maintain_order: bool) -> None:
    assert_frame_equal(
        pl.DataFrame(
            {
                "a": [1, 2, 3],
            }
        )
        .group_by("a", maintain_order=maintain_order)
        .agg(pl.lit([1]).gather([0, 0, 0])),
        pl.DataFrame({"a": [1, 2, 3], "literal": [[[1], [1], [1]]] * 3}),
        check_row_order=maintain_order,
    )


def test_get_window_with_filtered_empty_groups_23029() -> None:
    # https://github.com/pola-rs/polars/issues/23029
    df = pl.DataFrame(
        {
            "group": [1, 1, 2, 2, 3, 3],
            "value": [10, 20, 30, 40, 50, 60],
            "filter_condition": [False, True, False, False, True, True],
        }
    )

    result = df.with_columns(
        get_first=(
            pl.col("value")
            .filter(pl.col("filter_condition"))
            .get(0, null_on_oob=True)
            .over("group")
        ),
        first_value=(
            pl.col("value").filter(pl.col("filter_condition")).first().over("group")
        ),
    )

    assert_series_equal(
        result["get_first"],
        result["first_value"],
        check_names=False,
    )

    # And the concrete expected values are:
    expected = pl.DataFrame(
        {
            "group": [1, 1, 2, 2, 3, 3],
            "value": [10, 20, 30, 40, 50, 60],
            "filter_condition": [False, True, False, False, True, True],
            "get_first": [20, 20, None, None, 50, 50],
            "first_value": [20, 20, None, None, 50, 50],
        }
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("idx_dtype", [pl.Int64, pl.UInt64, pl.Int128, pl.UInt128])
def test_get_typed_index_null_on_oob_true(idx_dtype: pl.DataType) -> None:
    # OOB typed index with null_on_oob=True -> null, for multiple integer dtypes.
    df = pl.DataFrame({"value": [1, 2, 10]})

    out = df.select(v=pl.col("value").get(pl.lit(5, dtype=idx_dtype), null_on_oob=True))

    assert out["v"].to_list() == [None]


@pytest.mark.parametrize("idx_dtype", [pl.Int64, pl.UInt64, pl.Int128, pl.UInt128])
def test_get_typed_index_null_on_oob_false_raises(idx_dtype: pl.DataType) -> None:
    # OOB typed index with null_on_oob=False -> OutOfBoundsError, for multiple dtypes.
    df = pl.DataFrame({"value": [10, 11]})

    with pytest.raises(OutOfBoundsError, match="gather indices are out of bounds"):
        df.select(pl.col("value").get(pl.lit(5, dtype=idx_dtype), null_on_oob=False))


@pytest.mark.parametrize("idx_dtype", [pl.Int64, pl.UInt64, pl.Int128, pl.UInt128])
def test_get_typed_index_default_raises_out_of_bounds(idx_dtype: pl.DataType) -> None:
    # Default behavior (null_on_oob omitted) should behave like null_on_oob=False
    df = pl.DataFrame({"value": [10, 11]})

    with pytest.raises(OutOfBoundsError, match="gather indices are out of bounds"):
        df.select(pl.col("value").get(pl.lit(5, dtype=idx_dtype)))


def test_expr_gather_null_on_oob() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = df.select(pl.col("a").gather([0, 1, 10], null_on_oob=True))
    assert result["a"].to_list() == [1, 2, None]


def test_series_gather_null_on_oob() -> None:
    s = pl.Series("a", [1, 2, 3])

    result = s.gather([0, 1, 10], null_on_oob=True)
    assert result.to_list() == [1, 2, None]


@pytest.mark.parametrize("value", [1, 1.5, "a", True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_gather_scalar_with_interleaved_nulls(
    value: int | float | str | bool, nulls_last: bool
) -> None:
    frame = pl.DataFrame({"x": [value]})
    result = frame.gather(pl.Series([None, 0, None], dtype=pl.UInt32))
    values = [value, None, None] if nulls_last else [None, None, value]
    assert_frame_equal(
        result.sort("x", nulls_last=nulls_last), pl.DataFrame({"x": values})
    )


def test_gather_null_on_oob_group_by() -> None:
    lf = pl.LazyFrame({"g": [1, 2, 2], "x": [1.0, 2.0, 3.0]})
    # Indices outside the Int64 range.
    wide_idx = pl.Series([0, 2**63, -(2**63) - 1], dtype=pl.Int128)
    q = lf.group_by("g").agg(
        pos=pl.col("x").gather([0, 5], null_on_oob=True),
        neg=pl.col("x").gather([-1, -5], null_on_oob=True),
        big=pl.col("x").gather([0, 2**32], null_on_oob=True),
        wide=pl.col("x").gather(wide_idx, null_on_oob=True),
    )
    expected = pl.DataFrame(
        {
            "g": [1, 2],
            "pos": [[1.0, None], [2.0, None]],
            "neg": [[1.0, None], [3.0, None]],
            "big": [[1.0, None], [2.0, None]],
            "wide": [[1.0, None, None], [2.0, None, None]],
        }
    )
    for engine in ["in-memory", "streaming"]:
        assert_frame_equal(
            q.collect(engine=engine),  # type: ignore[call-overload]
            expected,
            check_row_order=False,
        )
    with pytest.raises(OutOfBoundsError):
        lf.group_by("g").agg(pl.col("x").gather([0, 5])).collect()

    assert_frame_equal(
        lf.select(pl.col("x").gather(wide_idx, null_on_oob=True)).collect(),
        pl.DataFrame({"x": [1.0, None, None]}),
    )
    with pytest.raises(OutOfBoundsError):
        lf.select(pl.col("x").gather(wide_idx)).collect()


@pytest.mark.parametrize(
    "idx",
    [
        [0, 2**32],
        [-1, 2**32],
        [-(2**32) - 1, 0],
        pl.Series([0, 2**32], dtype=pl.UInt64),
        pl.Series([0, 2**63, -(2**63) - 1], dtype=pl.Int128),
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_gather_group_by_large_index_raises(
    idx: list[int] | pl.Series, engine: EngineType
) -> None:
    q = (
        pl.LazyFrame({"g": [1, 1, 2], "x": [1, 2, 3]})
        .group_by("g")
        .agg(pl.col("x").gather(idx))
    )
    with pytest.raises(OutOfBoundsError):
        q.collect(engine=engine)


@pytest.mark.parametrize(
    "idx",
    [
        pl.Series([-1, 0], dtype=pl.Int8),
        pl.Series([-1, 0], dtype=pl.Int128),
        pl.Series([1, 0], dtype=pl.UInt64),
    ],
)
def test_gather_group_by_index_dtypes(idx: pl.Series) -> None:
    lf = pl.LazyFrame({"g": [1, 1, 2, 2], "x": [1, 2, 3, 4]})
    q = lf.group_by("g", maintain_order=True).agg(pl.col("x").gather(idx))
    expected = pl.DataFrame({"g": [1, 2], "x": [[2, 1], [4, 3]]})
    assert_frame_equal(q.collect(), expected)


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        (
            [["a", None], None, [], ["b", "a longer string value", "d"], None, ["e"]],
            pl.List(pl.String),
        ),
        ([[1, None], None, [], [2, 3, 4], None, [5]], pl.List(pl.Int64)),
        (
            [[[1], None, []], None, [], [[2, 3]], None, [None]],
            pl.List(pl.List(pl.Int64)),
        ),
        (
            [[{"x": 1}, None], None, [], [{"x": None}, {"x": 2}], None, [{"x": 3}]],
            pl.List(pl.Struct({"x": pl.Int64})),
        ),
        (
            [[[1, 2], None], None, [], [[3, 4]], None, [[5, 6]]],
            pl.List(pl.Array(pl.Int64, 2)),
        ),
        ([[True, None], None, [], [False, True], None, [True]], pl.List(pl.Boolean)),
        ([[None, None], None, [], [None], None, [None]], pl.List(pl.Null)),
        ([None] * 6, pl.List(pl.Int64)),
        (
            [[[1, None], None], None, [], [[None, 4]], None, [[5, 6]]],
            pl.List(pl.Array(pl.Int64, 2)),
        ),
        (
            [
                [["a", None], None],
                None,
                [],
                [["a longer string value", ""]],
                None,
                [["c", "d"]],
            ],
            pl.List(pl.Array(pl.String, 2)),
        ),
        (
            [
                [[{"x": 1}, None]],
                None,
                [],
                [[{"x": None}, {"x": 2}], None],
                None,
                [[{"x": 3}, {"x": 4}]],
            ],
            pl.List(pl.Array(pl.Struct({"x": pl.Int64}), 2)),
        ),
        (
            [
                [[[1, 2], [3, None]], None],
                None,
                [],
                [[[5, 6], [7, 8]]],
                None,
                [[None, [9, 10]]],
            ],
            pl.List(pl.Array(pl.Int64, (2, 2))),
        ),
        (
            [
                [[datetime(2026, 1, 1), None]],
                None,
                [],
                [None, [datetime(2026, 1, 2)] * 2],
                None,
                [[None, None]],
            ],
            pl.List(pl.Array(pl.Datetime("ms"), 2)),
        ),
        (
            [[[True, None, False]], None, [], [[False] * 3, None], None, [[True] * 3]],
            pl.List(pl.Array(pl.Boolean, 3)),
        ),
        ([[[], None], None, [], [[]], None, [[], []]], pl.List(pl.Array(pl.Int32, 0))),
        (
            [
                [{"a": [1, 2]}, None],
                None,
                [],
                [{"a": None}, {"a": [3, None]}],
                None,
                [{"a": [5, 6]}],
            ],
            pl.List(pl.Struct({"a": pl.Array(pl.Int64, 2)})),
        ),
        (
            [
                [list(range(128))],
                None,
                [],
                [list(range(128, 256))],
                None,
                [[None] * 128],
            ],
            pl.List(pl.Array(pl.Int32, 128)),
        ),
    ],
)
def test_gather_list_29993(data: list[Any], dtype: pl.DataType) -> None:
    s = pl.Series(data, dtype=dtype)
    idx = [5, None, 3, 3, 1, 0, 2, None, 4]
    expected = [None if i is None else data[i] for i in idx]
    assert s.gather(pl.Series(idx, dtype=pl.get_index_type())).to_list() == expected
    assert s[1:].gather([4, 0, 2]).to_list() == [data[5], data[1], data[3]]

    df = pl.DataFrame({"k": [0, 1, 0, 1, 0, 1], "a": s})
    out = df.group_by("k", maintain_order=True).agg("a")
    assert out["a"].to_list() == [data[0::2], data[1::2]]


@pytest.mark.parametrize(
    ("dtype", "leaf"),
    [
        (pl.Int64, lambda i: None if i % 7 == 0 else i),
        (pl.Boolean, lambda i: i % 3 == 0),
        (pl.String, lambda i: None if i % 5 == 0 else f"value number {i}"),
        (pl.Struct({"x": pl.Int64, "s": pl.String}), lambda i: {"x": i, "s": str(i)}),
        (pl.List(pl.Int64), lambda i: [i, None]),
        (pl.Array(pl.Int32, 2), lambda i: [i, None]),
        (pl.Array(pl.String, 2), lambda i: [str(i), None]),
    ],
)
def test_gather_list_wide_rows(dtype: pl.DataType, leaf: Callable[[int], Any]) -> None:
    # Wide rows of fixed-width values are copied whole instead of gathered per element.
    rows = [range(40), range(40, 40), range(40, 140), None, range(140, 440)]
    data = [None if r is None else [leaf(i) for i in r] for r in rows]
    s = pl.Series(data, dtype=pl.List(dtype))
    idx = [4, None, 2, 0, 3, 1, 4]
    expected = [None if i is None else data[i] for i in idx]
    assert s.gather(pl.Series(idx, dtype=pl.get_index_type())).to_list() == expected
    assert s[1:].gather([3, 0, 1]).to_list() == [data[4], data[1], data[2]]


def test_gather_list_null_rows_with_values() -> None:
    # Null row 1 still spans the value 3.
    s = pl.Series(
        pa.LargeListArray.from_arrays(
            pa.array([0, 2, 3, 5], pa.int64()),
            pa.array([1, 2, 3, 4, 5]),
            mask=pa.array([False, True, False]),
        )
    )
    assert s.to_arrow().offsets.to_pylist() == [0, 2, 3, 5]
    expected = pl.Series([[4, 5], None, [1, 2], None])
    assert_series_equal(s.gather([2, 1, 0, 1]), expected)


def test_gather_list_of_2d_array_null_hash() -> None:
    # A null 2-D array element must hash the same after a gather.
    s = pl.Series([[None, [[1, 2], [3, 4]]]], dtype=pl.List(pl.Array(pl.Int64, (2, 2))))
    assert pl.concat([s, s.gather([0])]).n_unique() == 1


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        ([[1, 2], None, [3, 4], [5, 6]], pl.Array(pl.Int64, 2)),
        ([[1, None, 3, 4], None, [5, 6, None, 8], [None] * 4], pl.Array(pl.Int64, 4)),
        (
            [["a", None], None, ["a longer string value", ""], [None, "b"]],
            pl.Array(pl.String, 2),
        ),
        (
            [[{"x": 1}, None], None, [{"x": None}, {"x": 2}], [{"x": 3}, {"x": 4}]],
            pl.Array(pl.Struct({"x": pl.Int64}), 2),
        ),
        (
            [[[1, 2], [3, None]], None, [None, [5, 6]], [[7, 8], [9, 10]]],
            pl.Array(pl.Int64, (2, 2)),
        ),
        ([[], None, [], []], pl.Array(pl.Int32, 0)),
        ([[True, None], None, [False, True], [None, None]], pl.Array(pl.Boolean, 2)),
    ],
)
def test_gather_array_rows(data: list[Any], dtype: pl.DataType) -> None:
    s = pl.Series(data, dtype=dtype)
    idx = [3, None, 1, 0, 0, 2, None]
    out = s.gather(pl.Series(idx, dtype=pl.get_index_type()))
    assert out.dtype == dtype
    assert out.to_list() == [None if i is None else data[i] for i in idx]
    assert s[1:].gather([2, 0, 1]).to_list() == [data[3], data[1], data[2]]
    # A sliced index validity bitmap.
    sliced_idx = pl.Series([None, 3, None, 1, 0], dtype=pl.get_index_type())[1:]
    assert s.gather(sliced_idx).to_list() == [data[3], None, data[1], data[0]]


def test_gather_list_of_array_hidden_values() -> None:
    # The null array element hides the valid values 9, 9.
    arr = pa.FixedSizeListArray.from_arrays(
        pa.array([9, 9, 1, 2, 3, 4], pa.int64()), 2, mask=pa.array([True, False, False])
    )
    s = pl.Series(
        "a", pa.LargeListArray.from_arrays(pa.array([0, 2, 3], pa.int64()), arr)
    )
    out = s.gather([1, 0, 0])
    assert out.to_list() == [[[3, 4]], [None, [1, 2]], [None, [1, 2]]]
    assert pl.concat([s, out]).n_unique() == 2
    assert out.gather([1, 0]).eq_missing(s).all()
    assert out.to_frame().group_by("a").len().height == 2

    buf = io.BytesIO()
    out.to_frame().write_ipc(buf)
    buf.seek(0)
    assert_series_equal(pl.read_ipc(buf).to_series(), out)


@pytest.mark.parametrize(
    "child",
    [
        pa.array(["x", "y", "a", "b"]),
        pa.array([[9], [9, 9], [1], [2]]),
        pa.array([{"f": 9}, {"f": 9}, {"f": 1}, {"f": 2}]),
        pa.array([9, 9, 1, None]),
        pa.array([True, True, False, None]),
    ],
)
def test_gather_array_null_rows_get_null_values(child: pa.Array) -> None:
    # Like a null index, a null row gives null values, not the ones stored under it.
    arr = pa.FixedSizeListArray.from_arrays(child, 2, mask=pa.array([True, False]))
    s = pl.Series(arr)
    assert s.to_arrow().values.is_null().to_pylist()[:2] == [False, False]
    out = s.gather(pl.Series([1, 0, None], dtype=pl.get_index_type()))
    assert out.to_list() == [s.to_list()[1], None, None]
    assert out.to_arrow().values.is_null().to_pylist()[2:] == [True] * 4
