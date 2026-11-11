from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

import polars as pl
from polars._typing import AsofJoinStrategy
from polars.datatypes.group import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
)
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric.strategies.core import column, dataframes

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import (
        AsofJoinStrategy,
        JoinBuildSide,
        JoinStrategy,
        MaintainOrderJoin,
    )
    from tests.conftest import PlMonkeyPatch


def test_streaming_full_outer_joins() -> None:
    n = 100
    dfa = pl.DataFrame(
        {
            "a": np.random.randint(0, 40, n),
            "idx": np.arange(0, n),
        }
    )

    n = 100
    dfb = pl.DataFrame(
        {
            "a": np.random.randint(0, 40, n),
            "idx": np.arange(0, n),
        }
    )

    join_strategies: list[tuple[JoinStrategy, bool]] = [
        ("full", False),
        ("full", True),
    ]
    for how, coalesce in join_strategies:
        q = (
            dfa.lazy()
            .join(dfb.lazy(), on="a", how=how, coalesce=coalesce)
            .sort(["idx"])
        )
        a = q.collect(engine="streaming")
        b = q.collect(engine="in-memory")
        assert_frame_equal(a, b, check_row_order=False)


def test_streaming_joins() -> None:
    n = 100
    dfa = pd.DataFrame(
        {
            "a": np.random.randint(0, 40, n),
            "b": np.arange(0, n),
        }
    )

    n = 100
    dfb = pd.DataFrame(
        {
            "a": np.random.randint(0, 40, n),
            "b": np.arange(0, n),
        }
    )
    dfa_pl = pl.from_pandas(dfa).sort("a")
    dfb_pl = pl.from_pandas(dfb)

    join_strategies: list[Literal["inner", "left"]] = ["inner", "left"]
    for how in join_strategies:
        pd_result = dfa.merge(dfb, on="a", how=how)
        pd_result.columns = pd.Index(["a", "b", "b_right"])

        pl_result = (
            dfa_pl.lazy()
            .join(dfb_pl.lazy(), on="a", how=how)
            .sort(["a", "b", "b_right"])
            .collect(engine="streaming")
        )

        a = (
            pl.from_pandas(pd_result)
            .with_columns(pl.all().cast(int))
            .sort(["a", "b", "b_right"])
        )
        assert_frame_equal(a, pl_result, check_dtypes=False)

        pd_result = dfa.merge(dfb, on=["a", "b"], how=how)

        pl_result = (
            dfa_pl.lazy()
            .join(dfb_pl.lazy(), on=["a", "b"], how=how)
            .sort(["a", "b"])
            .collect(engine="streaming")
        )

        # we cast to integer because pandas joins creates floats
        a = pl.from_pandas(pd_result).with_columns(pl.all().cast(int)).sort(["a", "b"])
        assert_frame_equal(a, pl_result, check_dtypes=False)


def test_streaming_cross_join_empty() -> None:
    df1 = pl.LazyFrame(data={"col1": ["a"]})

    df2 = pl.LazyFrame(
        data={"col1": []},
        schema={"col1": str},
    )

    out = df1.join(df2, how="cross").collect(engine="streaming")
    assert out.shape == (0, 2)
    assert out.columns == ["col1", "col1_right"]


@pytest.mark.parametrize("how", ["cross", "range"])
def test_streaming_join_empty_build_does_not_read_probe(how: str) -> None:
    seen = []

    def count(s: pl.Series) -> pl.Series:
        seen.append(len(s))
        return s

    probe = (
        pl.LazyFrame({"x": range(1000)})
        .with_columns(
            pl.col("x").map_batches(count, is_elementwise=True, return_dtype=pl.Int64)
        )
        .filter(pl.col("x") < 0)
    )
    empty = pl.LazyFrame({"a": pl.Series([], dtype=pl.Int64)})
    if how == "cross":
        q = empty.join(probe, how="cross")
    else:
        q = empty.join_where(probe, pl.col("a") < pl.col("x"))
    assert q.collect(engine="streaming").height == 0
    assert seen == []


@pytest.mark.parametrize("how", ["inner", "semi", "anti"])
def test_streaming_join_empty_side_stops_sampling_empty_morsels(
    how: JoinStrategy, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "1")
    seen = []

    def empty(df: pl.DataFrame) -> pl.DataFrame:
        seen.append(df.height)
        return df.clear()

    left = pl.LazyFrame(schema={"a": pl.Int64})
    right = pl.LazyFrame({"a": range(2000)}).map_batches(empty, streamable=True)
    left.join(right, on="a", how=how).collect(engine="streaming")
    assert sum(seen) < 2000


def test_streaming_join_head_does_not_read_later_probe_rows() -> None:
    # The second input of the probe side raises when it is read.
    failing = (
        pl.LazyFrame({"s": ["x", "y"]})
        .group_by("s")
        .agg()
        .select(pl.col("s").cast(pl.Int64).alias("b"))
    )
    probe = pl.concat([pl.LazyFrame({"b": [2]}), failing])
    # A build side over several phases, so the probe side is read while building.
    build = pl.concat([pl.LazyFrame({"a": [2]}), pl.LazyFrame({"a": [1]})])
    for q in [
        build.join(probe, how="cross", maintain_order="right_left"),
        build.join_where(probe, pl.col("a") <= pl.col("b")),
    ]:
        out = q.head(1).collect(engine="streaming")
        assert out["b"].to_list() == [2]


def test_streaming_cross_join_nullable_struct_probe() -> None:
    # Each probe row is repeated once per build row, its validity included.
    build = pl.LazyFrame({"a": [1, 2, 3]})
    probe = pl.LazyFrame({"s": [{"x": 1}, None, {"x": 3}, None]})
    expected = pl.DataFrame(
        {
            "a": [1, 2, 3] * 4,
            "s": [s for s in [{"x": 1}, None, {"x": 3}, None] for _ in range(3)],
        }
    )
    out = build.join(probe, how="cross").collect(engine="streaming")
    assert_frame_equal(out, expected, check_row_order=False)


def test_streaming_join_rechunk_12498() -> None:
    rows = pl.int_range(0, 2)

    a = pl.select(A=rows).lazy()
    b = pl.select(B=rows).lazy()

    q = a.join(b, how="cross")
    assert q.collect(engine="streaming").sort(["B", "A"]).to_dict(as_series=False) == {
        "A": [0, 1, 0, 1],
        "B": [0, 0, 1, 1],
    }


@pytest.mark.parametrize("maintain_order", [False, True])
def test_join_null_matches(maintain_order: bool) -> None:
    # null values in joins should never find a match.
    df_a = pl.LazyFrame(
        {
            "idx_a": [0, 1, 2],
            "a": [None, 1, 2],
        }
    )

    df_b = pl.LazyFrame(
        {
            "idx_b": [0, 1, 2, 3],
            "a": [None, 2, 1, None],
        }
    )
    # Semi
    assert_series_equal(
        df_a.join(
            df_b,
            on="a",
            how="semi",
            nulls_equal=True,
            maintain_order="left" if maintain_order else "none",
        ).collect()["idx_a"],
        pl.Series("idx_a", [0, 1, 2]),
        check_order=maintain_order,
    )
    assert_series_equal(
        df_a.join(
            df_b,
            on="a",
            how="semi",
            nulls_equal=False,
            maintain_order="left" if maintain_order else "none",
        ).collect()["idx_a"],
        pl.Series("idx_a", [1, 2]),
        check_order=maintain_order,
    )

    # Inner
    expected = pl.DataFrame({"idx_a": [2, 1], "a": [2, 1], "idx_b": [1, 2]})
    assert_frame_equal(
        df_a.join(
            df_b,
            on="a",
            how="inner",
            maintain_order="right" if maintain_order else "none",
        ).collect(),
        expected,
        check_row_order=maintain_order,
    )

    # Left outer
    expected = pl.DataFrame(
        {"idx_a": [0, 1, 2], "a": [None, 1, 2], "idx_b": [None, 2, 1]}
    )
    assert_frame_equal(
        df_a.join(
            df_b,
            on="a",
            how="left",
            maintain_order="left" if maintain_order else "none",
        ).collect(),
        expected,
        check_row_order=maintain_order,
    )
    # Full outer
    expected = pl.DataFrame(
        {
            "idx_a": [None, 2, 1, None, 0],
            "a": [None, 2, 1, None, None],
            "idx_b": [0, 1, 2, 3, None],
            "a_right": [None, 2, 1, None, None],
        }
    )
    assert_frame_equal(
        df_a.join(
            df_b,
            on="a",
            how="full",
            maintain_order="right" if maintain_order else "none",
        ).collect(),
        expected,
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_join_null_matches_multiple_keys(streaming: bool) -> None:
    df_a = pl.LazyFrame(
        {
            "a": [None, 1, 2],
            "idx": [0, 1, 2],
        }
    )

    df_b = pl.LazyFrame(
        {
            "a": [None, 2, 1, None, 1],
            "idx": [0, 1, 2, 3, 1],
            "c": [10, 20, 30, 40, 50],
        }
    )

    expected = pl.DataFrame({"a": [1], "idx": [1], "c": [50]})
    assert_frame_equal(
        df_a.join(df_b, on=["a", "idx"], how="inner").collect(
            engine="streaming" if streaming else "in-memory"
        ),
        expected,
        check_row_order=False,
    )
    expected = pl.DataFrame(
        {"a": [None, 1, 2], "idx": [0, 1, 2], "c": [None, 50, None]}
    )
    assert_frame_equal(
        df_a.join(df_b, on=["a", "idx"], how="left").collect(
            engine="streaming" if streaming else "in-memory"
        ),
        expected,
        check_row_order=False,
    )

    expected = pl.DataFrame(
        {
            "a": [None, None, None, None, None, 1, 2],
            "idx": [None, None, None, None, 0, 1, 2],
            "a_right": [None, 2, 1, None, None, 1, None],
            "idx_right": [0, 1, 2, 3, None, 1, None],
            "c": [10, 20, 30, 40, None, 50, None],
        }
    )
    assert_frame_equal(
        df_a.join(df_b, on=["a", "idx"], how="full").sort("a").collect(),
        expected,
        check_row_order=False,
    )


def test_streaming_join_and_union() -> None:
    a = pl.LazyFrame({"a": [1, 2]})

    b = pl.LazyFrame({"a": [1, 2, 4, 8]})

    c = a.join(b, on="a", maintain_order="left_right")
    # The join node latest ensures that the dispatcher
    # needs to replace placeholders in unions.
    q = pl.concat([a, b, c])

    out = q.collect(engine="streaming")
    assert_frame_equal(out, q.collect(engine="in-memory"))
    assert out.to_series().to_list() == [1, 2, 1, 2, 4, 8, 1, 2]


def test_non_coalescing_streaming_left_join() -> None:
    df1 = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    df2 = pl.LazyFrame({"a": [1, 2], "c": ["j", "i"]})

    q = df1.join(df2, on="a", how="left", coalesce=False)
    assert_frame_equal(
        q.collect(engine="streaming"),
        pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["a", "b", "c"],
                "a_right": [1, 2, None],
                "c": ["j", "i", None],
            }
        ),
        check_row_order=False,
    )


@pytest.mark.write_disk
def test_streaming_outer_join_partial_flush(tmp_path: Path) -> None:
    data = {
        "value_at": [datetime(2024, i + 1, 1) for i in range(6)],
        "value": list(range(6)),
    }

    parquet_path = tmp_path / "data.parquet"
    pl.DataFrame(data=data).write_parquet(parquet_path)

    other_parquet_path = tmp_path / "data2.parquet"
    pl.DataFrame(data=data).write_parquet(other_parquet_path)

    lf1 = pl.scan_parquet(other_parquet_path)
    lf2 = pl.scan_parquet(parquet_path)

    join_cols = set(lf1.collect_schema()).intersection(set(lf2.collect_schema()))
    final_lf = lf1.join(lf2, on=list(join_cols), how="full", coalesce=True)

    assert_frame_equal(
        final_lf.collect(engine="streaming"),
        pl.DataFrame(
            {
                "value_at": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 2, 1, 0, 0),
                    datetime(2024, 3, 1, 0, 0),
                    datetime(2024, 4, 1, 0, 0),
                    datetime(2024, 5, 1, 0, 0),
                    datetime(2024, 6, 1, 0, 0),
                ],
                "value": [0, 1, 2, 3, 4, 5],
            }
        ),
        check_row_order=False,
    )


def test_flush_join_and_operation_19040() -> None:
    df_A = pl.LazyFrame({"K": [True, False], "A": [1, 1]})

    df_B = pl.LazyFrame({"K": [True], "B": [1]})

    df_C = pl.LazyFrame({"K": [True], "C": [1]})

    q = (
        df_A.join(df_B, how="full", on=["K"], coalesce=True)
        .join(df_C, how="full", on=["K"], coalesce=True)
        .with_columns(B=pl.col("B"))
        .sort("K")
    )
    assert q.collect(engine="streaming").to_dict(as_series=False) == {
        "K": [False, True],
        "A": [1, 1],
        "B": [None, 1],
        "C": [None, 1],
    }


def test_full_coalesce_join_and_rename_15583() -> None:
    df1 = pl.LazyFrame({"a": [1, 2, 3]})
    df2 = pl.LazyFrame({"a": [3, 4, 5]})

    result = (
        df1.join(df2, on="a", how="full", coalesce=True)
        .select(pl.all().name.map(lambda c: c.upper()))
        .sort("A")
        .collect(engine="streaming")
    )
    assert result["A"].to_list() == [1, 2, 3, 4, 5]


def test_invert_order_full_join_22295() -> None:
    lf = pl.LazyFrame(
        {
            "value_at": [datetime(2024, i + 1, 1) for i in range(6)],
            "value": list(range(6)),
        }
    )
    lf.join(lf, on=["value", "value_at"], how="full", coalesce=True).collect(
        engine="streaming"
    )


def test_cross_join_with_literal_column_25544() -> None:
    df0 = pl.LazyFrame({"c0": [0]})
    df1 = pl.LazyFrame({"c0": [1]})

    result = df0.join(
        df1.select(pl.col("c0")).with_columns(pl.lit(1)),
        on=True,  # type: ignore[arg-type]
    ).select("c0")

    in_memory_result = result.collect(engine="in-memory")
    streaming_result = result.collect(engine="streaming")

    assert_frame_equal(streaming_result, in_memory_result)
    assert streaming_result.item() == 0


@pytest.mark.parametrize("on", [["key"], ["key", "key_ext"]])
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("coalesce", [None, True, False])
@pytest.mark.parametrize("maintain_order", ["none", "left_right", "right_left"])
@given(data=st.data())
@settings(max_examples=10)
def test_merge_join(
    on: list[str],
    how: JoinStrategy,
    descending: bool,
    nulls_last: bool,
    nulls_equal: bool,
    coalesce: bool | None,
    maintain_order: MaintainOrderJoin,
    data: st.DataObject,
) -> None:
    check_row_order = maintain_order in {"left_right", "right_left"}

    df_st = dataframes(min_cols=len(on), max_cols=len(on), allowed_dtypes=[pl.Int16])
    left_df = data.draw(df_st)
    right_df = data.draw(df_st)

    left = left_df.rename(dict(zip(left_df.columns, ["key", "key_ext"], strict=False)))
    right = right_df.rename(
        dict(zip(right_df.columns, ["key", "key_ext"], strict=False))
    )

    def df_sorted(df: pl.DataFrame) -> pl.LazyFrame:
        return (
            df.lazy()
            .sort(
                *on,
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=True,
                multithreaded=False,
            )
            .set_sorted(on, descending=descending, nulls_last=nulls_last)
        )

    q = df_sorted(left).join(
        df_sorted(right),
        on=on,
        how=how,
        nulls_equal=nulls_equal,
        coalesce=coalesce,
        maintain_order=maintain_order,
    )
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    expected = q.collect(engine="in-memory")
    actual = q.collect(engine="streaming")

    assert "merge-join" in dot, "merge-join not used in plan"
    assert_frame_equal(actual, expected, check_row_order=check_row_order)


def test_merge_join_ignores_projected_out_dataframe_columns() -> None:
    left = pl.LazyFrame({"key": [10, 11]}).with_row_index()
    right = pl.LazyFrame({"unused": [20, 21]}).with_row_index().select("index")
    q = left.join(right, on="index")
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)

    assert "merge-join" in dot, "merge-join not used in plan"
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("keys", "dtype"),
    [
        ([False, True, False], pl.Boolean),
        ([1, 3, 2], pl.Int8),
        ([1, 3, 2], pl.Int16),
        ([1, 3, 2], pl.Int32),
        ([1, 3, 2], pl.Int64),
        ([1, 3, 2], pl.Int128),
        ([1, 3, 2], pl.UInt8),
        ([1, 3, 2], pl.UInt16),
        ([1, 3, 2], pl.UInt32),
        ([1, 3, 2], pl.UInt64),
        ([1, 3, 2], pl.UInt128),
        ([1.0, 3.0, 2.0], pl.Float16),
        ([1.0, 3.0, 2.0], pl.Float32),
        ([1.0, 3.0, 2.0], pl.Float64),
        (["a", "b", "c"], pl.String),
        ([b"a", b"b", b"c"], pl.Binary),
        ([datetime(2024, 1, x) for x in [1, 3, 2]], pl.Date),
        ([datetime(2024, 1, x, 12, 0) for x in [1, 3, 2]], pl.Time),
        ([datetime(2024, 1, x, 12, 0) for x in [1, 3, 2]], pl.Datetime),
        ([timedelta(days=x) for x in [1, 3, 2]], pl.Duration),
        ([1, 3, 2], pl.Decimal),
        ([pl.Null, pl.Null, pl.Null], pl.Null),
        (["a", "c", "b"], pl.Enum(["a", "b", "c"])),
        (["a", "c", "b"], pl.Categorical),
    ],
)
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_join_dtypes(
    keys: list[Any], dtype: pl.DataType, how: JoinStrategy, nulls_equal: bool
) -> None:
    df_left = pl.DataFrame({"key": pl.Series("key", keys[:2], dtype=dtype)})
    df_right = pl.DataFrame({"key": pl.Series("key", keys[2:], dtype=dtype)})

    def df_sorted(df: pl.DataFrame) -> pl.LazyFrame:
        return (
            df.lazy()
            .sort(
                "key",
                maintain_order=True,
                multithreaded=False,
            )
            .set_sorted("key")
        )

    q_hashjoin = df_left.lazy().join(
        df_right.lazy(),
        on="key",
        how=how,
        nulls_equal=nulls_equal,
        maintain_order="none",
    )
    dot = q_hashjoin.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True
    )
    expected = q_hashjoin.collect(engine="in-memory")
    actual = q_hashjoin.collect(engine="streaming")
    assert "equi-join" in dot, "hash-join not used in plan"
    assert_frame_equal(actual, expected, check_row_order=False)

    q_mergejoin = df_sorted(df_left).join(
        df_sorted(df_right),
        on="key",
        how=how,
        nulls_equal=nulls_equal,
        maintain_order="none",
    )
    dot = q_mergejoin.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True
    )
    expected = q_mergejoin.collect(engine="in-memory")
    actual = q_mergejoin.collect(engine="streaming")
    assert "merge-join" in dot, "merge-join not used in plan"
    assert_frame_equal(actual, expected, check_row_order=False)


def test_merge_join_exprs() -> None:
    left = pl.LazyFrame(
        {
            "key": ["", "a", "c"],
            "key_ext": [1, 2, 3],
            "value": [1, 2, 3],
        }
    ).set_sorted("key", "key_ext")
    right = pl.LazyFrame(
        {
            "key": ["", "a", "b"],
            "key_ext": [3, 2, 3],
            "value": [4, 5, 6],
        }
    ).set_sorted("key", "key_ext")

    q = left.join(
        right,
        left_on="key",
        right_on=pl.concat_str(pl.col("key"), ignore_nulls=False),
        how="full",
        maintain_order="left_right",
    )
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "merge-join" in dot, "merge-join not used in plan"
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("maintain_order", ["none", "left_right", "right_left"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("on", [["key"], ["key", "key_ext"]])
@pytest.mark.parametrize("chunk_size", [4, None])
def test_merge_join_many_partitions(
    how: JoinStrategy,
    maintain_order: MaintainOrderJoin,
    descending: bool,
    nulls_last: bool,
    nulls_equal: bool,
    on: list[str],
    chunk_size: int | None,
) -> None:
    # Small morsels split each window into many partitions, and the long runs of 0 and
    # null are split again on the build side. With the default morsel size the window is
    # a single partition. The left keys go on past the right ones.
    left_keys = (
        [None] * 20
        + [0] * 25
        + [1, 2, 2, 3, 5, 5, 5, 8]
        + [9] * 12
        + list(range(11, 60))
    )
    right_keys = [None] * 18 + [0] * 20 + [2, 3, 3, 4, 5, 9, 9, 10, 11, 11]

    def sorted_lf(keys: list[int | None], name: str) -> pl.LazyFrame:
        return (
            pl.LazyFrame(
                {"key": keys, name: range(len(keys))},
                schema_overrides={"key": pl.Int64},
            )
            .with_columns(
                key_ext=pl.when(pl.col(name) % 7 == 0)
                .then(None)
                .otherwise(pl.col("key") % 3)
            )
            .sort(on, descending=descending, nulls_last=nulls_last, maintain_order=True)
            .set_sorted(on, descending=descending, nulls_last=nulls_last)
        )

    q = sorted_lf(left_keys, "a").join(
        sorted_lf(right_keys, "b"),
        on=on,
        how=how,
        nulls_equal=nulls_equal,
        maintain_order=maintain_order,
    )
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "merge-join" in dot, "merge-join not used in plan"

    expected = q.collect(engine="in-memory")
    with pl.Config(streaming_chunk_size=chunk_size):
        actual = q.collect(engine="streaming")
        actual_all, actual_head = pl.collect_all([q, q.head(3)], engine="streaming")
    check_row_order = maintain_order != "none"
    assert_frame_equal(actual, expected, check_row_order=check_row_order)
    assert_frame_equal(actual_all, expected, check_row_order=check_row_order)
    assert actual_head.height == 3
    if check_row_order:
        assert_frame_equal(actual_head, expected.head(3))


@pytest.mark.slow
@pytest.mark.may_fail_cloud  # reason: timing
@pytest.mark.parametrize("null_run_side", ["left", "right"])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_merge_join_null_run_linear_time(null_run_side: str, nulls_last: bool) -> None:
    # A run of nulls, shorter than a morsel, next to many other keys must not be scanned
    # again for every one of those keys. The outliers sit beyond the keys, on the far
    # side from the nulls.
    n = 90_000
    keys = pl.Series(range(n), dtype=pl.Int64)
    outliers = [-(10**12), -(10**12) + 1] if nulls_last else [10**12, 10**12 + 1]
    null_run = pl.Series([None] * n + outliers, dtype=pl.Int64)
    left, right = (null_run, keys) if null_run_side == "left" else (keys, null_run)

    q = (
        pl.LazyFrame({"key": left})
        .sort("key", nulls_last=nulls_last)
        .join(
            pl.LazyFrame({"key": right}).sort("key", nulls_last=nulls_last),
            on="key",
            nulls_equal=True,
            maintain_order="left_right",
        )
    )
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "merge-join" in dot, "merge-join not used in plan"

    t0 = time.time()
    result = q.collect(engine="streaming")
    t1 = time.time()
    assert result.height == 0
    assert (t1 - t0) < 2


@pytest.mark.parametrize("df_level_hint", [False, True])
@pytest.mark.parametrize("left_descending", [False, True])
@pytest.mark.parametrize("right_descending", [False, True])
@pytest.mark.parametrize("left_nulls_last", [False, True])
@pytest.mark.parametrize("right_nulls_last", [False, True])
def test_merge_join_applicable(
    df_level_hint: bool,
    left_descending: bool,
    right_descending: bool,
    left_nulls_last: bool,
    right_nulls_last: bool,
) -> None:
    def make_set_sorted_lf(
        df_level_hint: bool, descending: bool, nulls_last: bool
    ) -> pl.LazyFrame:
        lf = pl.LazyFrame({"key": [1]})
        if df_level_hint:
            return lf.set_sorted("key", descending=descending, nulls_last=nulls_last)
        else:
            return lf.with_columns(
                pl.col("key").set_sorted(descending=descending, nulls_last=nulls_last)
            )

    left = make_set_sorted_lf(df_level_hint, left_descending, left_nulls_last)
    right = make_set_sorted_lf(df_level_hint, right_descending, right_nulls_last)
    q = left.join(right, on="key", how="full", maintain_order="left_right")
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    if (
        left_descending == right_descending
        and left_nulls_last == right_nulls_last is not None
    ):
        assert "merge-join" in dot
    else:
        assert "merge-join" not in dot
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize(
    ("how", "maintain_order", "coalesce"),
    [
        *(
            (how, maintain_order, coalesce)
            for how in ["inner", "left", "semi", "anti"]
            for maintain_order in ["left", "left_right"]
            for coalesce in [False, True]
        ),
        ("inner", "right", False),
        ("inner", "right_left", False),
        ("right", "right", False),
        ("right", "right", True),
        ("right", "right_left", True),
    ],
)
def test_merge_join_after_order_maintaining_join(
    how: JoinStrategy, maintain_order: MaintainOrderJoin, coalesce: bool
) -> None:
    # https://github.com/pola-rs/polars/issues/29841
    sorted_lf = pl.LazyFrame({"a": [0, 1, 1, 3], "b": [1, 2, 3, 4]}).set_sorted("a")
    other = pl.LazyFrame({"a": [1, 0, 2, 1], "c": [5, 6, 7, 8]})
    if maintain_order.startswith("left"):
        lf = sorted_lf.join(
            other, on="a", how=how, maintain_order=maintain_order, coalesce=coalesce
        )
    else:
        lf = other.join(
            sorted_lf, on="a", how=how, maintain_order=maintain_order, coalesce=coalesce
        )
    key = "a_right" if not maintain_order.startswith("left") and not coalesce else "a"
    lf = lf.select(pl.col(key).alias("key"), pl.exclude("a", "a_right"))

    q = lf.join(
        pl.LazyFrame({"key": [0, 1], "d": [9, 10]}).set_sorted("key"),
        on="key",
        how="left",
        maintain_order="left",
    )
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "merge-join" in dot
    out = q.collect(engine="streaming")
    expected = q.collect(engine="in-memory")
    # Several matches of one row may come in any order.
    assert_series_equal(out["key"], expected["key"])
    assert_frame_equal(out, expected, check_row_order=False)


_SORTED_LF = pl.LazyFrame({"a": [0, 1]}).set_sorted("a")
_UNSORTED_LF = pl.LazyFrame({"a": [1, 0, 2]})


@pytest.mark.parametrize(
    ("lf", "maintain_order"),
    [
        pytest.param(
            _SORTED_LF.join(_UNSORTED_LF, on="a", how="left"),
            "left",
            id="join-without-maintain-order",
        ),
        pytest.param(
            _SORTED_LF.join(
                _UNSORTED_LF, on="a", how="full", maintain_order="left", coalesce=True
            ),
            "left",
            id="full-join-appends-unmatched-right-rows",
        ),
        pytest.param(
            _UNSORTED_LF.join(_SORTED_LF, on="a", how="inner", maintain_order="left"),
            "left",
            id="join-keeps-order-of-unsorted-side",
        ),
        pytest.param(
            _SORTED_LF.join(_UNSORTED_LF, on="a", how="left", maintain_order="left"),
            "none",
            id="order-not-observed",
        ),
    ],
)
def test_no_merge_join_after_join(
    lf: pl.LazyFrame, maintain_order: MaintainOrderJoin
) -> None:
    q = lf.join(_SORTED_LF, on="a", how="left", maintain_order=maintain_order)
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "merge-join" not in dot


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("allow_exact_matches", [False, True])
@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize(
    "key_dtypes",
    [
        FLOAT_DTYPES,
        INTEGER_DTYPES,
        {pl.String, pl.Binary},
        {pl.Boolean},
        {pl.Date},
        {
            pl.Datetime("ms"),
            pl.Datetime("us"),
            pl.Datetime("ns"),
        },
        {
            pl.Datetime("ms", time_zone="Europe/Amsterdam"),
            pl.Datetime("us", time_zone="Europe/Amsterdam"),
            pl.Datetime("ns", time_zone="Europe/Amsterdam"),
        },
        {pl.Time},
        {pl.Duration("ms"), pl.Duration("us"), pl.Duration("ns")},
    ],
)
@pytest.mark.parametrize("n_groups", [0, 1, 2])
@given(data=st.data())
@settings(max_examples=20)
def test_streaming_asof_join(
    data: st.DataObject,
    strategy: AsofJoinStrategy,
    allow_exact_matches: bool,
    coalesce: bool,
    key_dtypes: set[pl.DataType],
    n_groups: int,
) -> None:
    GROUP_DTYPES = [
        *INTEGER_DTYPES,
        pl.Boolean,
        pl.String,
        pl.Binary,
        pl.Categorical,
    ]

    if key_dtypes & {pl.String, pl.Binary, pl.Boolean} and strategy == "nearest":
        pytest.skip(
            "asof join with string/binary/bool does not support 'nearest' strategy"
        )

    group_col_names = [f"group{i}" for i in range(n_groups)]
    group_cols = [
        column(name=name, dtype=data.draw(st.sampled_from(GROUP_DTYPES)))
        for name in group_col_names
    ]

    val_dtype = data.draw(st.sampled_from(list(key_dtypes)))
    df_st = dataframes(
        min_cols=1,
        max_cols=1,
        allowed_dtypes=[val_dtype],
        allow_time_zones=False,
        include_cols=group_cols,
    )
    left_df = data.draw(df_st)
    right_df = data.draw(df_st)

    left = left_df.rename({"col0": "key"}).sort("key").with_row_index().lazy()
    right = right_df.rename({"col0": "key"}).sort("key").with_row_index().lazy()

    if n_groups > 0:
        descending = data.draw(st.booleans())
        nulls_last = data.draw(st.booleans())
        left = left.sort(
            group_col_names,
            maintain_order=True,
            descending=descending,
            nulls_last=nulls_last,
        )
        right = right.sort(
            group_col_names,
            maintain_order=True,
            descending=descending,
            nulls_last=nulls_last,
        )
        q = left.join_asof(
            right,
            on="key",
            by=group_col_names,
            strategy=strategy,
            allow_exact_matches=allow_exact_matches,
            coalesce=coalesce,
        )
    else:
        q = left.join_asof(
            right,
            on="key",
            strategy=strategy,
            allow_exact_matches=allow_exact_matches,
            coalesce=coalesce,
        )

    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "asof-join" in plan

    expected = q.collect(engine="in-memory")
    actual = q.collect(engine="streaming")
    assert_frame_equal(actual, expected)


def test_streaming_merge_join_send_port_done_27547() -> None:
    left = pl.LazyFrame({"key": [1, 2, 3], "left_payload": ["a", "b", "c"]})
    right = pl.LazyFrame({"key": [1, 2, 3], "right_payload": ["d", "e", "f"]})
    q = (
        left.set_sorted("key")
        .join(
            right.set_sorted("key"),
            on="key",
            how="inner",
        )
        .head(2)
    )
    expected = q.collect(engine="in-memory")
    actual = q.collect(engine="streaming")
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("threads", ["1", "2", "4"])
@pytest.mark.parametrize("morsel_size", ["1", "4", "100000"])
@pytest.mark.parametrize(
    ("how", "maintain_order", "swap"),
    [
        ("right", "left_right", False),
        ("full", "left_right", False),
        ("left", "right_left", True),
        ("full", "right_left", True),
    ],
)
def test_streaming_join_unmatched_build_order_29362(
    plmonkeypatch: PlMonkeyPatch,
    threads: str,
    morsel_size: str,
    how: JoinStrategy,
    maintain_order: MaintainOrderJoin,
    swap: bool,
) -> None:
    plmonkeypatch.setenv("POLARS_MAX_THREADS", threads)
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", morsel_size)

    n = 200
    # Interleaved duplicate keys, nulls, and matches so that unmatched rows
    # from the build side come from many hash-table slots and partitions.
    keys = [None if i % 7 == 0 else i % 5 for i in range(n)]
    build = pl.LazyFrame({"k": keys, "row": range(n)})
    probe = pl.LazyFrame({"k": [1], "probe_row": [0]})

    left, right = (build, probe) if swap else (probe, build)
    q = left.join(right, on="k", how=how, maintain_order=maintain_order)

    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "equi-join" in plan

    expected = q.collect(engine="in-memory")
    unmatched = expected.filter(pl.col("probe_row").is_null())["row"].to_list()
    assert unmatched == sorted(unmatched)
    assert_frame_equal(q.collect(engine="streaming"), expected)


@pytest.mark.parametrize("threads", ["1", "2"])
def test_streaming_join_unmatched_build_order_string_keys_29362(
    plmonkeypatch: PlMonkeyPatch, threads: str
) -> None:
    plmonkeypatch.setenv("POLARS_MAX_THREADS", threads)
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "3")

    n = 100
    right = pl.LazyFrame(
        {
            "k": [None if i % 11 == 0 else f"k{i % 6}" for i in range(n)],
            "k2": [i % 3 for i in range(n)],
            "row": range(n),
        }
    )
    left = pl.LazyFrame({"k": ["k1", "k4"], "k2": [1, 0], "lv": ["a", "b"]})
    q = left.join(right, on=["k", "k2"], how="full", maintain_order="left_right")

    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "equi-join" in plan

    expected = q.collect(engine="in-memory")
    assert_frame_equal(q.collect(engine="streaming"), expected)


@pytest.mark.parametrize(
    ("left_keys", "how", "coalesce"),
    [
        ([1], "right", True),
        ([1], "full", True),
        ([42], "right", False),
        ([], "right", False),
    ],
)
def test_streaming_join_unmatched_build_order_coalesce_29362(
    plmonkeypatch: PlMonkeyPatch,
    left_keys: list[int],
    how: JoinStrategy,
    coalesce: bool,
) -> None:
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "2")
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "100000")

    left = pl.LazyFrame({"k": left_keys}, {"k": pl.Int64})
    right = pl.LazyFrame({"k": [1, 2, 3, 2, 3, 2, 3, 2, 3], "row": range(9)})
    q = left.join(
        right, on="k", how=how, maintain_order="left_right", coalesce=coalesce
    )
    expected = q.collect(engine="in-memory")
    assert expected["row"].to_list() == list(range(9))
    assert_frame_equal(q.collect(engine="streaming"), expected)


@pytest.mark.parametrize(
    ("left_on", "merge_join"),
    [
        # Clipping can make different values equal, the next key is then not sorted.
        ([pl.col("a").clip(0, 10), "b"], False),
        ([pl.col("a").clip(0, 10) * 3, "b"], False),
        (["a", pl.col("b").clip(0, 10)], True),
    ],
)
def test_merge_join_clipped_key(left_on: list[pl.Expr | str], merge_join: bool) -> None:
    left = pl.LazyFrame({"a": [-2, -1, 0, 0], "b": [2, 1, -2, -1]}).set_sorted("a", "b")
    right = pl.LazyFrame({"a": [0, 0], "b": [0, 1]}).set_sorted("a", "b")
    q = left.join(right, left_on=left_on, right_on=["a", "b"])

    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert ("merge-join" in str(dot)) == merge_join
    expected = q.collect(engine="in-memory")
    assert expected.height > 0
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


@pytest.mark.parametrize("how", ["inner", "semi"])
def test_streaming_join_empty_build_side_after_repeat(how: JoinStrategy) -> None:
    # The join stops reading its probe side, which `repeat` must pass on.
    lf = pl.LazyFrame({"a": [1, None, 3], "c": ["x", None, "y"]})
    empty = pl.LazyFrame({"k": pl.Series([], dtype=pl.String)})
    q = lf.filter(pl.repeat(True, pl.col("a").len())).join(
        empty, left_on="c", right_on="k", how=how
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_merge_join_set_sorted_key() -> None:
    # "c" does not come from the sorted column "a", so "b" is not sorted within it.
    left = pl.LazyFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [0, 0, 0, 0]}
    ).set_sorted("a", "b")
    right = pl.LazyFrame({"c": [0, 0], "b": [1, 2]}).set_sorted("c", "b")
    q = left.join(right, left_on=[pl.col("c").set_sorted(), "b"], right_on=["c", "b"])
    expected = q.collect(engine="in-memory")
    assert expected.height == 4
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


def _join_keys(k: pl.Expr, key_type: str) -> dict[str, pl.Expr]:
    # Each key type uses a different kind of hash table.
    if key_type == "int":
        return {"k": k}
    if key_type == "str":
        return {"k": k.cast(pl.String)}
    if key_type == "int, int":
        return {"k": k, "k2": k % 3}
    struct = pl.when(k.is_not_null()).then(pl.struct(a=k, b=k % 3))
    if key_type == "struct":
        return {"k": struct}
    assert key_type == "struct, int"
    return {"k": struct, "k2": k % 3}


@pytest.mark.parametrize(
    "key_type", ["int", "str", "int, int", "struct", "struct, int"]
)
@pytest.mark.parametrize("how", ["left", "right", "full"])
@pytest.mark.parametrize("build_side", ["force_left", "force_right"])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("maintain_order", ["none", "left_right", "right_left"])
def test_streaming_join_unmatched_build_rows(
    key_type: str,
    how: JoinStrategy,
    build_side: JoinBuildSide,
    nulls_equal: bool,
    maintain_order: MaintainOrderJoin,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # Small morsels so the unmatched rows of a table are emitted in parts.
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "7")
    # Enough rows that a partition of the table spans several 64-row words.
    i = pl.int_range(3000)
    keys = _join_keys(pl.when(i % 7 != 0).then(i % 50), key_type)
    left = pl.select([e.alias(n) for n, e in keys.items()], v=i).lazy()
    i = pl.int_range(60)
    # Both sides have keys without a match.
    keys = _join_keys(pl.when(i % 5 != 0).then(i % 25 + 30), key_type)
    right = pl.select([e.alias(n) for n, e in keys.items()], w=i).lazy()
    q = left.join(
        right,
        on=list(keys),
        how=how,
        build_side=build_side,
        nulls_equal=nulls_equal,
        maintain_order=maintain_order,
    )
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=maintain_order != "none",
    )


@pytest.mark.parametrize(
    "key_type", ["int", "str", "int, int", "struct", "struct, int"]
)
def test_streaming_join_unmatched_repeated_key_morsels(
    key_type: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    # One key with many unmatched rows is not split into a morsel per row.
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "100000")
    keys = _join_keys(pl.lit(1, pl.Int64), key_type)
    left = pl.select([e.alias(n) for n, e in keys.items()], v=pl.int_range(3000))
    keys = _join_keys(pl.lit(2, pl.Int64), key_type)
    right = pl.select([e.alias(n) for n, e in keys.items()], w=0)
    q = left.lazy().join(
        right.lazy(), on=list(keys), how="left", build_side="force_left"
    )
    out = q.collect(engine="streaming")
    assert_frame_equal(out, q.collect(engine="in-memory"), check_row_order=False)
    # At most a morsel per pipeline, there is one pipeline per thread.
    assert out.n_chunks() <= pl.thread_pool_size()


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_streaming_join_keeps_height_of_input_without_columns(
    how: Literal["inner", "left", "right", "full"],
) -> None:
    # Only `id` is selected, so the left input keeps no columns.
    left = pl.LazyFrame({"x": [10, 20, 30, 40]})
    right = pl.LazyFrame({"id": [1, 2], "f": [True, False]})
    q = left.join_where(right, pl.col("f"), how=how).select("id")
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=False,
    )


SPILL_ENV = {
    "POLARS_OOC_MEMORY_BUDGET_MB": "0",
    "POLARS_OOC_SPILL_MIN_BYTES": "1",
    "POLARS_OOC_DRIFT_THRESHOLD": "0",
}


@pytest.fixture
def spill_join(plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    """Spills the build side and the samples, in several morsels and batches."""
    for name, value in SPILL_ENV.items():
        plmonkeypatch.setenv(name, value)
    plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "16")
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "40")
    plmonkeypatch.setenv("POLARS_JOIN_BATCH_BYTES", "1024")


# The number of rows, the key modulus and the null key interval of each input of
# `_spill_join_inputs`.
_SPILL_JOIN_LEFT = (150, 50, 7)
_SPILL_JOIN_RIGHT = (120, 45, 11)


def _spill_join_input(
    key_type: str, name: str, num_rows: int, modulo: int, null_every: int
) -> pl.LazyFrame:
    i = pl.int_range(num_rows)
    keys = _join_keys(pl.when(i % null_every != 0).then(i % modulo), key_type)
    return pl.select(*(e.alias(n) for n, e in keys.items()), i.alias(name)).lazy()


def _spill_join_inputs(key_type: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    left = _spill_join_input(key_type, "l", *_SPILL_JOIN_LEFT)
    right = _spill_join_input(key_type, "r", *_SPILL_JOIN_RIGHT)
    return left, right


def _spill_join_keys(n: int, modulo: int, null_every: int) -> list[int | None]:
    """The keys of `_spill_join_inputs` as integers."""
    return [None if i % null_every == 0 else i % modulo for i in range(n)]


def _expected_join(how: str, nulls_equal: bool, *, left_first: bool) -> pl.DataFrame:
    """
    The `(l, r)` rows of a join of `_spill_join_inputs`.

    Each row of the first side in order with its matches in order, then the
    unmatched rows of the other side in order.
    """
    left = _spill_join_keys(*_SPILL_JOIN_LEFT)
    right = _spill_join_keys(*_SPILL_JOIN_RIGHT)
    return _expected_join_of_keys(left, right, how, nulls_equal, left_first=left_first)


def _expected_join_of_keys(
    left: list[int | None],
    right: list[int | None],
    how: str,
    nulls_equal: bool,
    *,
    left_first: bool = True,
) -> pl.DataFrame:
    """The `(l, r)` rows of a join of rows with the keys `left` and `right`."""
    first, other = (left, right) if left_first else (right, left)
    keep_first = how == "full" or how == ("left" if left_first else "right")
    keep_other = how == "full" or how == ("right" if left_first else "left")

    other_rows: dict[int | None, list[int]] = {}
    for j, k in enumerate(other):
        if k is not None or nulls_equal:
            other_rows.setdefault(k, []).append(j)
    rows: list[tuple[int | None, int | None]] = []
    matched = set()
    for i, k in enumerate(first):
        js = other_rows.get(k, [])
        matched.update(js)
        rows.extend((i, j) for j in js)
        if not js and keep_first:
            rows.append((i, None))
    if keep_other:
        rows.extend((None, j) for j in range(len(other)) if j not in matched)

    if not left_first:
        rows = [(i, j) for j, i in rows]
    return pl.DataFrame(rows, schema={"l": pl.Int64, "r": pl.Int64}, orient="row")


@pytest.mark.write_disk
@pytest.mark.parametrize("threads", ["1", None])
@pytest.mark.parametrize("key_type", ["int", "str", "int, int", "struct"])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize(
    ("maintain_order", "build_side"),
    [
        ("none", "auto"),
        ("none", "force_left"),
        ("none", "force_right"),
        ("left", "auto"),
        ("right", "auto"),
        ("left_right", "auto"),
        ("right_left", "auto"),
    ],
)
@pytest.mark.usefixtures("spill_join")
def test_streaming_join_spilled_build(
    maintain_order: MaintainOrderJoin,
    build_side: JoinBuildSide,
    how: JoinStrategy,
    nulls_equal: bool,
    key_type: str,
    threads: str | None,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    if threads is not None:
        plmonkeypatch.setenv("POLARS_MAX_THREADS", threads)
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    left, right = _spill_join_inputs(key_type)
    on = list(_join_keys(pl.lit(0), key_type))
    q = left.join(
        right,
        on=on,
        how=how,
        nulls_equal=nulls_equal,
        maintain_order=maintain_order,
        build_side=build_side,
    ).select("l", "r")

    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    [num_batches] = re.findall(r"\[equi-join\]: building .* (\d+) batches", err)
    assert int(num_batches) > 1

    left_first = maintain_order in ("none", "left", "left_right")
    expected = _expected_join(how, nulls_equal, left_first=left_first)
    in_order = maintain_order in ("left_right", "right_left")
    assert_frame_equal(out, expected, check_row_order=in_order)
    if maintain_order in ("left", "right"):
        name = maintain_order[0]
        assert_series_equal(out[name], expected[name])


@pytest.mark.write_disk
@pytest.mark.parametrize("threads", ["1", None])
@pytest.mark.parametrize("key_type", ["int", "str", "int, int", "struct"])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("how", ["semi", "anti"])
@pytest.mark.usefixtures("spill_join")
def test_streaming_semi_anti_join_spilled_samples(
    how: JoinStrategy,
    nulls_equal: bool,
    key_type: str,
    threads: str | None,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    if threads is not None:
        plmonkeypatch.setenv("POLARS_MAX_THREADS", threads)
    left, right = _spill_join_inputs(key_type)
    on = list(_join_keys(pl.lit(0), key_type))
    q = left.join(right, on=on, how=how, nulls_equal=nulls_equal).select("l")

    out = q.collect(engine="streaming")
    full = _expected_join("left", nulls_equal, left_first=True)
    matched = full.filter(pl.col("r").is_not_null())["l"].unique()
    expected = (
        full.select("l")
        .unique()
        .filter(pl.col("l").is_in(matched.implode()) == (how == "semi"))
    )
    assert_frame_equal(out, expected, check_row_order=False)


_SPILLED_MODE = pytest.param(
    "spilled", marks=[pytest.mark.write_disk, pytest.mark.slow]
)


@pytest.fixture(params=["in_memory", "grace", "hybrid", "one_partition", _SPILLED_MODE])
def join_mode(
    request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> str:
    """Makes small joins take the paths of large ones."""
    mode = str(request.param)
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "8")
    plmonkeypatch.setenv("POLARS_JOIN_BATCH_BYTES", "1024")
    if mode != "in_memory":
        num_partitions = 1 if mode == "one_partition" else 4 * pl.thread_pool_size() + 3
        in_memory = num_partitions // 2 if mode in ("hybrid", "spilled") else 0
        plmonkeypatch.setenv("POLARS_JOIN_NUM_PARTITIONS", str(num_partitions))
        plmonkeypatch.setenv("POLARS_JOIN_IN_MEMORY_PARTITIONS", str(in_memory))
    if mode == "spilled":
        for name, value in SPILL_ENV.items():
            plmonkeypatch.setenv(name, value)
        plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))
    return mode


def _modes(*modes: object) -> pytest.MarkDecorator:
    return pytest.mark.parametrize("join_mode", modes, indirect=True)


# The in-memory cases are covered by the spilled build tests.
_OUT_OF_CORE_MODES = _modes("grace", "hybrid", "one_partition", _SPILLED_MODE)


@_OUT_OF_CORE_MODES
@pytest.mark.parametrize("threads", ["1", None])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("build_side", ["force_left", "force_right"])
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_streaming_join_out_of_core(
    join_mode: str,
    how: JoinStrategy,
    build_side: JoinBuildSide,
    nulls_equal: bool,
    threads: str | None,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    if threads is not None:
        plmonkeypatch.setenv("POLARS_MAX_THREADS", threads)
    left, right = _spill_join_inputs("int")
    q = left.join(
        right, on="k", how=how, nulls_equal=nulls_equal, build_side=build_side
    ).select("l", "r")
    expected = _expected_join(how, nulls_equal, left_first=True)
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


@_OUT_OF_CORE_MODES
@pytest.mark.parametrize("key_type", ["str", "int, int", "struct"])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_streaming_join_out_of_core_key_types(
    join_mode: str, key_type: str, nulls_equal: bool
) -> None:
    left, right = _spill_join_inputs(key_type)
    on = list(_join_keys(pl.lit(0), key_type))
    q = left.join(
        right, on=on, how="full", nulls_equal=nulls_equal, build_side="force_left"
    ).select("l", "r")
    expected = _expected_join("full", nulls_equal, left_first=True)
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize("build_side", ["force_left", "force_right"])
def test_streaming_join_out_of_core_full_coalesce(
    join_mode: str, build_side: JoinBuildSide, coalesce: bool
) -> None:
    left, right = _spill_join_inputs("int")
    q = left.join(right, on="k", how="full", coalesce=coalesce, build_side=build_side)
    left_keys = _spill_join_keys(*_SPILL_JOIN_LEFT)
    right_keys = _spill_join_keys(*_SPILL_JOIN_RIGHT)
    rows = _expected_join("full", False, left_first=True).rows()
    k = [None if i is None else left_keys[i] for i, _ in rows]
    k_right = [None if j is None else right_keys[j] for _, j in rows]
    columns: dict[str, list[int | None]] = {"k": k}
    if coalesce:
        columns["k"] = [
            a if a is not None else b for a, b in zip(k, k_right, strict=True)
        ]
    columns["l"] = [i for i, _ in rows]
    if not coalesce:
        columns["k_right"] = k_right
    columns["r"] = [j for _, j in rows]
    expected = pl.DataFrame(columns, schema=dict.fromkeys(columns, pl.Int64))
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


@pytest.mark.parametrize(
    "case",
    ["empty_left", "empty_right", "null_keys", "few_left_keys", "few_right_keys"],
)
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_streaming_join_out_of_core_sides(
    join_mode: str, how: JoinStrategy, case: str
) -> None:
    # With few keys on one side, most stored partitions have no rows of that side.
    left_keys: list[int | None] = [i % 40 for i in range(150)]
    right_keys: list[int | None] = [i % 50 for i in range(120)]
    if case == "empty_left":
        left_keys = []
    elif case == "empty_right":
        right_keys = []
    elif case == "null_keys":
        left_keys = [None] * len(left_keys)
    elif case == "few_left_keys":
        left_keys = [i % 3 for i in range(150)]
    else:
        right_keys = [i % 3 for i in range(120)]
    left = pl.LazyFrame(
        {"k": left_keys, "l": range(len(left_keys))},
        schema={"k": pl.Int64, "l": pl.Int64},
    )
    right = pl.LazyFrame(
        {"k": right_keys, "r": range(len(right_keys))},
        schema={"k": pl.Int64, "r": pl.Int64},
    )
    q = left.join(right, on="k", how=how, build_side="force_right").select("l", "r")
    expected = _expected_join_of_keys(left_keys, right_keys, how, False)
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


def test_streaming_join_out_of_core_fused_predicate(join_mode: str) -> None:
    left, right = _spill_join_inputs("int")
    q = left.join(right, on="k", build_side="force_right").filter(
        pl.col("l") < pl.col("r")
    )
    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert plan is not None
    assert "fused predicate:" in plan
    expected = _expected_join("inner", False, left_first=True).filter(
        pl.col("l") < pl.col("r")
    )
    out = q.select("l", "r").collect(engine="streaming")
    assert_frame_equal(out, expected, check_row_order=False)


def test_streaming_join_out_of_core_slice(join_mode: str) -> None:
    left, right = _spill_join_inputs("int")
    q = left.join(right, on="k", build_side="force_right").select("l", "r")
    expected = set(_expected_join("inner", False, left_first=True).rows())
    # In the grace and one_partition modes every row is sent while flushing.
    for out, height in [
        (q.slice(20, 30).collect(engine="streaming"), 30),
        (q.head(5).collect(engine="streaming"), 5),
    ]:
        assert out.schema == pl.Schema({"l": pl.Int64, "r": pl.Int64})
        assert out.height == height
        assert set(out.rows()) <= expected


@pytest.mark.parametrize("how", ["inner", "full"])
def test_streaming_join_out_of_core_into_sampling_join(
    join_mode: str,
    how: JoinStrategy,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The second join stops its inputs after sampling 10 rows, during the probe or
    # the flush of the first join, which resumes in a later phase. With one
    # pipeline, few rows are sent before the stop is seen.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10")
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "1")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    left, right = _spill_join_inputs("int")
    other = pl.LazyFrame({"l": range(150), "x": range(0, 300, 2)})
    q = (
        left.join(right, on="k", how=how, build_side="force_right")
        .select("l", "r")
        .join(other, on="l")
    )
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    if how == "inner" and join_mode in ("grace", "one_partition"):
        # All rows are sent while flushing.
        assert "[equi-join]: resuming" in err
    expected = (
        _expected_join(how, False, left_first=True)
        .filter(pl.col("l").is_not_null())
        .with_columns(x=2 * pl.col("l"))
    )
    assert_frame_equal(out, expected, check_row_order=False)


_JOIN_BUILD_LINE = re.compile(
    r"\[equi-join\]: building (\d+) partitions \((\d+) in memory\).* "
    r"memory grant of (\d+) bytes \((\d+) wanted\)"
)


def _join_builds(err: str) -> list[tuple[int, int, int, int]]:
    """
    Return what each equi-join in the verbose output chose, largest grant first.

    That is its partitions, its partitions in memory, its grant and the grant it
    wanted.
    """
    builds = [
        (int(p), int(r), int(g), int(w)) for p, r, g, w in _JOIN_BUILD_LINE.findall(err)
    ]
    return sorted(builds, key=lambda b: -b[2])


def _join_small_budget(plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    """Set a memory budget of 1 MB, of which a join keeps up to 500 kB in tables."""
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_OOC_MEMORY_BUDGET_MB", "1")
    plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "4")


def _wide_dimension(key: str, value: str, n: int) -> pl.LazyFrame:
    """
    Return `n` rows with a 400 byte string key, so that few rows give a large table.

    Row `i` has key `i` and value `2 * i`.
    """
    i = pl.int_range(n)
    return pl.select(_wide_key(i).alias(key), (2 * i).alias(value)).lazy()


def _wide_key(i: pl.Expr) -> pl.Expr:
    return i.cast(pl.String).str.pad_start(400, "0")


@pytest.mark.write_disk
def test_streaming_join_memory_grant_chained(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _join_small_budget(plmonkeypatch, tmp_path)
    n = 500
    i = pl.int_range(2 * n)
    fact = pl.select(
        k1=_wide_key(i % n), k2=_wide_key((7 * i) % n), f=i.cast(pl.Int64)
    ).lazy()
    # Both joins hold their tables while the fact rows are probed.
    q = (
        fact.join(_wide_dimension("k1", "a", n), on="k1", build_side="force_right")
        .join(_wide_dimension("k2", "b", n), on="k2", build_side="force_right")
        .select("f", "a", "b")
    )
    capfd.readouterr()
    out = q.collect(engine="streaming")
    first, second = _join_builds(capfd.readouterr().err)
    # The first gets all it wants and keeps all tables in memory. The second gets
    # two thirds of what the first left in the pool, too little for its tables.
    assert first[2] == first[3]
    assert first[0] == first[1]
    assert second[2] < second[3]
    assert second[1] < second[0]
    f = pl.int_range(2 * n)
    expected = pl.select(f=f, a=2 * (f % n), b=2 * ((7 * f) % n))
    assert_frame_equal(out, expected, check_row_order=False)


def test_streaming_join_memory_grant_fraction_zero(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_OOC_MEMORY_GRANT_FRACTION", "0")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    left, right = _spill_join_inputs("int")
    q = left.join(right, on="k", how="full", build_side="force_right").select("l", "r")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    [(num_partitions, in_memory, granted, _)] = _join_builds(capfd.readouterr().err)
    assert granted == 0
    # Only the tables of partitions without build rows are kept in memory.
    assert in_memory < num_partitions
    expected = _expected_join("full", False, left_first=True)
    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.write_disk
def test_streaming_join_memory_grant_returned(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _join_small_budget(plmonkeypatch, tmp_path)
    n = 1000
    i = pl.int_range(n)
    fact = pl.select(k=_wide_key((7 * i) % n), f=i, s=i.cast(pl.String)).lazy()
    q = fact.join(_wide_dimension("k", "a", n), on="k", build_side="force_right")
    expected = pl.select(f=i, a=2 * ((7 * i) % n))
    capfd.readouterr()
    # The tables do not fit in the largest grant, so partitions are stored.
    for _ in range(2):
        out = q.select("f", "a").collect(engine="streaming")
        [(num_partitions, in_memory, granted, wanted)] = _join_builds(
            capfd.readouterr().err
        )
        assert (granted, wanted) == (500_000, 500_000)
        assert in_memory < num_partitions
        assert_frame_equal(out, expected, check_row_order=False)

    # Only the last fact row fails to parse, after the tables are built.
    failing = q.with_columns(
        pl.when(pl.col("f") < n - 1).then(pl.col("s")).otherwise(pl.lit("x"))
    ).select(pl.col("s").str.to_integer())
    with pytest.raises(pl.exceptions.ComputeError, match="strict integer parsing"):
        failing.collect(engine="streaming")
    assert [b[2] for b in _join_builds(capfd.readouterr().err)] == [500_000]

    out = q.select("f", "a").collect(engine="streaming")
    assert [b[2] for b in _join_builds(capfd.readouterr().err)] == [500_000]
    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.write_disk
def test_streaming_join_memory_grant_ordered(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _join_small_budget(plmonkeypatch, tmp_path)
    n = 1000
    i = pl.int_range(n)
    fact = pl.select(k=_wide_key((7 * i) % n), f=i).lazy()
    q = fact.join(
        _wide_dimension("k", "a", n), on="k", maintain_order="left_right"
    ).select("f", "a")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    [(num_partitions, in_memory, granted, wanted)] = _join_builds(
        capfd.readouterr().err
    )
    # Ordered joins ask for all their tables and keep them in memory with less.
    assert granted < wanted
    assert in_memory == num_partitions
    assert_frame_equal(out, pl.select(f=i, a=2 * ((7 * i) % n)))
