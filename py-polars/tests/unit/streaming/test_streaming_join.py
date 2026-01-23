from __future__ import annotations

import itertools
import typing
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import JoinStrategy, MaintainOrderJoin

pytestmark = pytest.mark.xdist_group("streaming")


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


@pytest.mark.slow
@pytest.mark.parametrize("on", [["key"], ["key", "key_ext"]])
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("nulls_equal", [False, True])
@pytest.mark.parametrize("coalesce", [None, True, False])
@pytest.mark.parametrize("maintain_order", ["none", "left_right", "right_left"])
@pytest.mark.parametrize("ideal_morsel_size", [1, 1000])
def test_merge_join(
    on: list[str],
    how: JoinStrategy,
    descending: bool,
    nulls_last: bool,
    nulls_equal: bool,
    coalesce: bool | None,
    maintain_order: MaintainOrderJoin,
    ideal_morsel_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_examples = 10
    key_value_set = pl.Series([None] * 5 + list(range(5)), dtype=pl.Int32)
    check_row_order = maintain_order in {"left_right", "right_left"}
    monkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", str(ideal_morsel_size))

    def sample_keys(height: int, seed: int) -> pl.Series:
        return key_value_set.sample(
            height, with_replacement=True, shuffle=True, seed=seed
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

    seed = 0
    for height, _ in itertools.product([0, 1, 5, 10], range(max_examples)):
        # Use random testing, because hypothesis does not work well with
        # monkeypatch.

        df_left = pl.DataFrame(
            {
                "key": sample_keys(height, seed),
                "key_ext": sample_keys(height, seed + 1),
            },
        ).with_row_index()
        df_right = pl.DataFrame(
            {
                "key": sample_keys(height, seed + 2),
                "key_ext": sample_keys(height, seed + 3),
            },
        ).with_row_index()
        seed += 4

        q = df_sorted(df_left).join(
            df_sorted(df_right),
            on=on,
            how=how,
            nulls_equal=nulls_equal,
            coalesce=coalesce,
            maintain_order=maintain_order,
        )
        dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
        expected = q.collect(engine="in-memory")
        actual = q.collect(engine="streaming")

        assert "merge-join" in typing.cast("str", dot), "merge-join not used in plan"
        assert_frame_equal(actual, expected, check_row_order=check_row_order)


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
    assert "equi-join" in typing.cast("str", dot), "hash-join not used in plan"
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
    assert "merge-join" in typing.cast("str", dot), "merge-join not used in plan"
    assert_frame_equal(actual, expected, check_row_order=False)


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_merge_join_exprs(ignore_nulls: bool) -> None:
    left = pl.LazyFrame(
        {
            "key": ["zzzaaa", "zzzaaaa", "zzzcaaa"],
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
        right_on=pl.concat_str(
            pl.lit("zzz"), pl.col("key"), pl.lit("aaa"), ignore_nulls=ignore_nulls
        ),
        how="full",
        maintain_order="none",
    )
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "merge-join" in typing.cast("str", dot), "merge-join not used in plan"
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("left_descending", [False, True])
@pytest.mark.parametrize("right_descending", [False, True])
@pytest.mark.parametrize("left_nulls_last", [False, True])
@pytest.mark.parametrize("right_nulls_last", [False, True])
def test_merge_join_applicable(
    left_descending: bool,
    right_descending: bool,
    left_nulls_last: bool,
    right_nulls_last: bool,
) -> None:
    left = pl.LazyFrame({"key": [1]}).set_sorted(
        "key", descending=left_descending, nulls_last=left_nulls_last
    )
    right = pl.LazyFrame({"key": [2]}).set_sorted(
        "key", descending=right_descending, nulls_last=right_nulls_last
    )
    q = left.join(right, on="key", how="full", maintain_order="left_right")
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    if (left_descending, left_nulls_last) == (right_descending, right_nulls_last):
        assert "merge-join" in typing.cast("str", dot)
    else:
        assert "merge-join" not in typing.cast("str", dot)
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))
