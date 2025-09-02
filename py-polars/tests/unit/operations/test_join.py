from __future__ import annotations

import typing
import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    SchemaError,
)
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import time_func

if TYPE_CHECKING:
    from polars._typing import JoinStrategy, PolarsDataType


def test_semi_anti_join() -> None:
    df_a = pl.DataFrame({"key": [1, 2, 3], "payload": ["f", "i", None]})

    df_b = pl.DataFrame({"key": [3, 4, 5, None]})

    assert df_a.join(df_b, on="key", how="anti").to_dict(as_series=False) == {
        "key": [1, 2],
        "payload": ["f", "i"],
    }
    assert df_a.join(df_b, on="key", how="semi").to_dict(as_series=False) == {
        "key": [3],
        "payload": [None],
    }

    # lazy
    result = df_a.lazy().join(df_b.lazy(), on="key", how="anti").collect()
    expected_values = {"key": [1, 2], "payload": ["f", "i"]}
    assert result.to_dict(as_series=False) == expected_values

    result = df_a.lazy().join(df_b.lazy(), on="key", how="semi").collect()
    expected_values = {"key": [3], "payload": [None]}
    assert result.to_dict(as_series=False) == expected_values

    df_a = pl.DataFrame(
        {"a": [1, 2, 3, 1], "b": ["a", "b", "c", "a"], "payload": [10, 20, 30, 40]}
    )

    df_b = pl.DataFrame({"a": [3, 3, 4, 5], "b": ["c", "c", "d", "e"]})

    assert df_a.join(df_b, on=["a", "b"], how="anti").to_dict(as_series=False) == {
        "a": [1, 2, 1],
        "b": ["a", "b", "a"],
        "payload": [10, 20, 40],
    }
    assert df_a.join(df_b, on=["a", "b"], how="semi").to_dict(as_series=False) == {
        "a": [3],
        "b": ["c"],
        "payload": [30],
    }


def test_join_same_cat_src() -> None:
    df = pl.DataFrame(
        data={"column": ["a", "a", "b"], "more": [1, 2, 3]},
        schema=[("column", pl.Categorical), ("more", pl.Int32)],
    )
    df_agg = df.group_by("column").agg(pl.col("more").mean())
    assert_frame_equal(
        df.join(df_agg, on="column"),
        pl.DataFrame(
            {
                "column": ["a", "a", "b"],
                "more": [1, 2, 3],
                "more_right": [1.5, 1.5, 3.0],
            },
            schema=[
                ("column", pl.Categorical),
                ("more", pl.Int32),
                ("more_right", pl.Float64),
            ],
        ),
        check_row_order=False,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_sorted_merge_joins(reverse: bool) -> None:
    n = 30
    df_a = pl.DataFrame({"a": np.sort(np.random.randint(0, n // 2, n))}).with_row_index(
        "row_a"
    )
    df_b = pl.DataFrame(
        {"a": np.sort(np.random.randint(0, n // 2, n // 2))}
    ).with_row_index("row_b")

    if reverse:
        df_a = df_a.select(pl.all().reverse())
        df_b = df_b.select(pl.all().reverse())

    join_strategies: list[JoinStrategy] = ["left", "inner"]
    for cast_to in [int, str, float]:
        for how in join_strategies:
            df_a_ = df_a.with_columns(pl.col("a").cast(cast_to))
            df_b_ = df_b.with_columns(pl.col("a").cast(cast_to))

            # hash join
            out_hash_join = df_a_.join(df_b_, on="a", how=how)

            # sorted merge join
            out_sorted_merge_join = df_a_.with_columns(
                pl.col("a").set_sorted(descending=reverse)
            ).join(
                df_b_.with_columns(pl.col("a").set_sorted(descending=reverse)),
                on="a",
                how=how,
            )

            assert_frame_equal(
                out_hash_join, out_sorted_merge_join, check_row_order=False
            )


def test_join_negative_integers() -> None:
    expected = pl.DataFrame({"a": [-6, -1, 0], "b": [-6, -1, 0]})
    df1 = pl.DataFrame(
        {
            "a": [-1, -6, -3, 0],
        }
    )

    df2 = pl.DataFrame(
        {
            "a": [-6, -1, -4, -2, 0],
            "b": [-6, -1, -4, -2, 0],
        }
    )

    for dt in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        assert_frame_equal(
            df1.with_columns(pl.all().cast(dt)).join(
                df2.with_columns(pl.all().cast(dt)), on="a", how="inner"
            ),
            expected.select(pl.all().cast(dt)),
            check_row_order=False,
        )


def test_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    other = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
    result = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [3, 4]})

    np.testing.assert_equal(
        df.join(other=other, on="a", maintain_order="left").to_numpy(),
        result.to_numpy(),
    )
    np.testing.assert_equal(
        df.lazy()
        .join(other=other.lazy(), on="a", maintain_order="left")
        .collect()
        .to_numpy(),
        result.to_numpy(),
    )


def test_deprecated_parameter_join_nulls() -> None:
    df = pl.DataFrame({"a": [1, None]})
    with pytest.deprecated_call(
        match=r"the argument `join_nulls` for `DataFrame.join` is deprecated. It was renamed to `nulls_equal`"
    ):
        result = df.join(df, on="a", join_nulls=True)  # type: ignore[call-arg]
    assert_frame_equal(result, df, check_row_order=False)


def test_join_on_expressions() -> None:
    df_a = pl.DataFrame({"a": [1, 2, 3]})

    df_b = pl.DataFrame({"b": [1, 4, 9, 9, 0]})

    assert_frame_equal(
        df_a.join(df_b, left_on=(pl.col("a") ** 2).cast(int), right_on=pl.col("b")),
        pl.DataFrame({"a": [1, 2, 3, 3], "b": [1, 4, 9, 9]}),
        check_row_order=False,
    )


def test_join_lazy_frame_on_expression() -> None:
    # Tests a lazy frame projection pushdown bug
    # https://github.com/pola-rs/polars/issues/19822

    df = pl.DataFrame(data={"a": [0, 1], "b": [2, 3]})

    lazy_join = (
        df.lazy()
        .join(df.lazy(), left_on=pl.coalesce("b", "a"), right_on="a")
        .select("a")
        .collect()
    )

    eager_join = df.join(df, left_on=pl.coalesce("b", "a"), right_on="a").select("a")

    assert lazy_join.shape == eager_join.shape


def test_right_join_schema_maintained_22516() -> None:
    df_left = pl.DataFrame({"number": [1]})
    df_right = pl.DataFrame({"invoice_number": [1]})
    eager_join = df_left.join(
        df_right, left_on="number", right_on="invoice_number", how="right"
    ).select(pl.len())

    lazy_join = (
        df_left.lazy()
        .join(df_right.lazy(), left_on="number", right_on="invoice_number", how="right")
        .select(pl.len())
        .collect()
    )

    assert lazy_join.item() == eager_join.item()


def test_join() -> None:
    df_left = pl.DataFrame(
        {
            "a": ["a", "b", "a", "z"],
            "b": [1, 2, 3, 4],
            "c": [6, 5, 4, 3],
        }
    )
    df_right = pl.DataFrame(
        {
            "a": ["b", "c", "b", "a"],
            "k": [0, 3, 9, 6],
            "c": [1, 0, 2, 1],
        }
    )

    joined = df_left.join(
        df_right, left_on="a", right_on="a", maintain_order="left_right"
    ).sort("a")
    assert_series_equal(joined["b"], pl.Series("b", [1, 3, 2, 2]))

    joined = df_left.join(
        df_right, left_on="a", right_on="a", how="left", maintain_order="left_right"
    ).sort("a")
    assert joined["c_right"].is_null().sum() == 1
    assert_series_equal(joined["b"], pl.Series("b", [1, 3, 2, 2, 4]))

    joined = df_left.join(df_right, left_on="a", right_on="a", how="full").sort("a")
    assert joined["c_right"].null_count() == 1
    assert joined["c"].null_count() == 1
    assert joined["b"].null_count() == 1
    assert joined["k"].null_count() == 1
    assert joined["a"].null_count() == 1

    # we need to pass in a column to join on, either by supplying `on`, or both
    # `left_on` and `right_on`
    with pytest.raises(ValueError):
        df_left.join(df_right)
    with pytest.raises(ValueError):
        df_left.join(df_right, right_on="a")
    with pytest.raises(ValueError):
        df_left.join(df_right, left_on="a")

    df_a = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})
    df_b = pl.DataFrame(
        {"foo": [1, 1, 1], "bar": ["a", "c", "c"], "ham": ["let", "var", "const"]}
    )

    # just check if join on multiple columns runs
    df_a.join(df_b, left_on=["a", "b"], right_on=["foo", "bar"])
    eager_join = df_a.join(df_b, left_on="a", right_on="foo")
    lazy_join = df_a.lazy().join(df_b.lazy(), left_on="a", right_on="foo").collect()

    cols = ["a", "b", "bar", "ham"]
    assert lazy_join.shape == eager_join.shape
    assert_frame_equal(lazy_join.sort(by=cols), eager_join.sort(by=cols))


def test_joins_dispatch() -> None:
    # this just flexes the dispatch a bit

    # don't change the data of this dataframe, this triggered:
    # https://github.com/pola-rs/polars/issues/1688
    dfa = pl.DataFrame(
        {
            "a": ["a", "b", "c", "a"],
            "b": [1, 2, 3, 1],
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-01"],
            "datetime": [13241324, 12341256, 12341234, 13241324],
        }
    ).with_columns(
        pl.col("date").str.strptime(pl.Date), pl.col("datetime").cast(pl.Datetime)
    )

    join_strategies: list[JoinStrategy] = ["left", "inner", "full"]
    for how in join_strategies:
        dfa.join(dfa, on=["a", "b", "date", "datetime"], how=how)
        dfa.join(dfa, on=["date", "datetime"], how=how)
        dfa.join(dfa, on=["date", "datetime", "a"], how=how)
        dfa.join(dfa, on=["date", "a"], how=how)
        dfa.join(dfa, on=["a", "datetime"], how=how)
        dfa.join(dfa, on=["date"], how=how)


def test_join_on_cast() -> None:
    df_a = (
        pl.DataFrame({"a": [-5, -2, 3, 3, 9, 10]})
        .with_row_index()
        .with_columns(pl.col("a").cast(pl.Int32))
    )

    df_b = pl.DataFrame({"a": [-2, -3, 3, 10]})

    assert_frame_equal(
        df_a.join(df_b, on=pl.col("a").cast(pl.Int64)),
        pl.DataFrame(
            {
                "index": [1, 2, 3, 5],
                "a": [-2, 3, 3, 10],
                "a_right": [-2, 3, 3, 10],
            }
        ),
        check_row_order=False,
        check_dtypes=False,
    )
    assert df_a.lazy().join(
        df_b.lazy(),
        on=pl.col("a").cast(pl.Int64),
        maintain_order="left",
    ).collect().to_dict(as_series=False) == {
        "index": [1, 2, 3, 5],
        "a": [-2, 3, 3, 10],
        "a_right": [-2, 3, 3, 10],
    }


def test_join_chunks_alignment_4720() -> None:
    # https://github.com/pola-rs/polars/issues/4720

    df1 = pl.DataFrame(
        {
            "index1": pl.arange(0, 2, eager=True),
            "index2": pl.arange(10, 12, eager=True),
        }
    )

    df2 = pl.DataFrame(
        {
            "index3": pl.arange(100, 102, eager=True),
        }
    )

    df3 = pl.DataFrame(
        {
            "index1": pl.arange(0, 2, eager=True),
            "index2": pl.arange(10, 12, eager=True),
            "index3": pl.arange(100, 102, eager=True),
        }
    )
    assert_frame_equal(
        df1.join(df2, how="cross").join(
            df3,
            on=["index1", "index2", "index3"],
            how="left",
        ),
        pl.DataFrame(
            {
                "index1": [0, 0, 1, 1],
                "index2": [10, 10, 11, 11],
                "index3": [100, 101, 100, 101],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df1.join(df2, how="cross").join(
            df3,
            on=["index3", "index1", "index2"],
            how="left",
        ),
        pl.DataFrame(
            {
                "index1": [0, 0, 1, 1],
                "index2": [10, 10, 11, 11],
                "index3": [100, 101, 100, 101],
            }
        ),
        check_row_order=False,
    )


def test_jit_sort_joins() -> None:
    n = 200
    # Explicitly specify numpy dtype because of different defaults on Windows
    dfa = pd.DataFrame(
        {
            "a": np.random.randint(0, 100, n, dtype=np.int64),
            "b": np.arange(0, n, dtype=np.int64),
        }
    )

    n = 40
    dfb = pd.DataFrame(
        {
            "a": np.random.randint(0, 100, n, dtype=np.int64),
            "b": np.arange(0, n, dtype=np.int64),
        }
    )
    dfa_pl = pl.from_pandas(dfa).sort("a")
    dfb_pl = pl.from_pandas(dfb)

    join_strategies: list[Literal["left", "inner"]] = ["left", "inner"]
    for how in join_strategies:
        pd_result = dfa.merge(dfb, on="a", how=how)
        pd_result.columns = pd.Index(["a", "b", "b_right"])

        # left key sorted right is not
        pl_result = dfa_pl.join(dfb_pl, on="a", how=how).sort(["a", "b", "b_right"])

        a = (
            pl.from_pandas(pd_result)
            .with_columns(pl.all().cast(int))
            .sort(["a", "b", "b_right"])
        )
        assert_frame_equal(a, pl_result)
        assert pl_result["a"].flags["SORTED_ASC"]

        # left key sorted right is not
        pd_result = dfb.merge(dfa, on="a", how=how)
        pd_result.columns = pd.Index(["a", "b", "b_right"])
        pl_result = dfb_pl.join(dfa_pl, on="a", how=how).sort(["a", "b", "b_right"])

        a = (
            pl.from_pandas(pd_result)
            .with_columns(pl.all().cast(int))
            .sort(["a", "b", "b_right"])
        )
        assert_frame_equal(a, pl_result)
        assert pl_result["a"].flags["SORTED_ASC"]


def test_join_panic_on_binary_expr_5915() -> None:
    df_a = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    df_b = pl.DataFrame({"b": [1, 4, 9, 9, 0]}).lazy()

    z = df_a.join(df_b, left_on=[(pl.col("a") + 1).cast(int)], right_on=[pl.col("b")])
    assert z.collect().to_dict(as_series=False) == {"a": [3], "b": [4]}


def test_semi_join_projection_pushdown_6423() -> None:
    df1 = pl.DataFrame({"x": [1]}).lazy()
    df2 = pl.DataFrame({"y": [1], "x": [1]}).lazy()

    assert (
        df1.join(df2, left_on="x", right_on="y", how="semi")
        .join(df2, left_on="x", right_on="y", how="semi")
        .select(["x"])
    ).collect().to_dict(as_series=False) == {"x": [1]}


def test_semi_join_projection_pushdown_6455() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 1, 2],
            "timestamp": [
                datetime(2022, 12, 11),
                datetime(2022, 12, 12),
                datetime(2022, 1, 1),
            ],
            "value": [1, 2, 4],
        }
    ).lazy()

    latest = df.group_by("id").agg(pl.col("timestamp").max())
    df = df.join(latest, on=["id", "timestamp"], how="semi")
    assert df.select(["id", "value"]).collect().to_dict(as_series=False) == {
        "id": [1, 2],
        "value": [2, 4],
    }


def test_update() -> None:
    df1 = pl.DataFrame(
        {
            "key1": [1, 2, 3, 4],
            "key2": [1, 2, 3, 4],
            "a": [1, 2, 3, 4],
            "b": [1, 2, 3, 4],
            "c": ["1", "2", "3", "4"],
            "d": [
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2023, 1, 4),
            ],
        }
    )

    df2 = pl.DataFrame(
        {
            "key1": [1, 2, 3, 4],
            "key2": [1, 2, 3, 5],
            "a": [1, 1, 1, 1],
            "b": [2, 2, 2, 2],
            "c": ["3", "3", "3", "3"],
            "d": [
                date(2023, 5, 5),
                date(2023, 5, 5),
                date(2023, 5, 5),
                date(2023, 5, 5),
            ],
        }
    )

    # update only on key1
    expected = pl.DataFrame(
        {
            "key1": [1, 2, 3, 4],
            "key2": [1, 2, 3, 5],
            "a": [1, 1, 1, 1],
            "b": [2, 2, 2, 2],
            "c": ["3", "3", "3", "3"],
            "d": [
                date(2023, 5, 5),
                date(2023, 5, 5),
                date(2023, 5, 5),
                date(2023, 5, 5),
            ],
        }
    )
    assert_frame_equal(df1.update(df2, on="key1"), expected)

    # update on key1 using different left/right names
    assert_frame_equal(
        df1.update(
            df2.rename({"key1": "key1b"}),
            left_on="key1",
            right_on="key1b",
        ),
        expected,
    )

    # update on key1 and key2. This should fail to match the last item.
    expected = pl.DataFrame(
        {
            "key1": [1, 2, 3, 4],
            "key2": [1, 2, 3, 4],
            "a": [1, 1, 1, 4],
            "b": [2, 2, 2, 4],
            "c": ["3", "3", "3", "4"],
            "d": [
                date(2023, 5, 5),
                date(2023, 5, 5),
                date(2023, 5, 5),
                date(2023, 1, 4),
            ],
        }
    )
    assert_frame_equal(df1.update(df2, on=["key1", "key2"]), expected)

    # update on key1 and key2 using different left/right names
    assert_frame_equal(
        df1.update(
            df2.rename({"key1": "key1b", "key2": "key2b"}),
            left_on=["key1", "key2"],
            right_on=["key1b", "key2b"],
        ),
        expected,
    )

    df = pl.DataFrame({"A": [1, 2, 3, 4], "B": [400, 500, 600, 700]})

    new_df = pl.DataFrame({"B": [4, None, 6], "C": [7, 8, 9]})

    assert df.update(new_df).to_dict(as_series=False) == {
        "A": [1, 2, 3, 4],
        "B": [4, 500, 6, 700],
    }
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [2, 3], "b": [8, 9]})

    assert df1.update(df2, on="a").to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [4, 8, 9],
    }

    a = pl.LazyFrame({"a": [1, 2, 3]})
    b = pl.LazyFrame({"b": [4, 5], "c": [3, 1]})
    c = a.update(b)

    assert_frame_equal(a, c)

    # check behaviour of 'how' param
    result = a.update(b, left_on="a", right_on="c")
    assert result.collect().to_series().to_list() == [1, 2, 3]

    result = a.update(b, how="inner", left_on="a", right_on="c")
    assert sorted(result.collect().to_series().to_list()) == [1, 3]

    result = a.update(b.rename({"b": "a"}), how="full", on="a")
    assert sorted(result.collect().to_series().sort().to_list()) == [1, 2, 3, 4, 5]

    # check behavior of include_nulls=True
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [400, 500, 600, 700],
        }
    )
    new_df = pl.DataFrame(
        {
            "B": [-66, None, -99],
            "C": [5, 3, 1],
        }
    )
    out = df.update(new_df, left_on="A", right_on="C", how="full", include_nulls=True)
    expected = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [-99, 500, None, 700, -66],
        }
    )
    assert_frame_equal(out, expected, check_row_order=False)

    # edge-case #11684
    x = pl.DataFrame({"a": [0, 1]})
    y = pl.DataFrame({"a": [2, 3]})
    assert sorted(x.update(y, on="a", how="full")["a"].to_list()) == [0, 1, 2, 3]

    # disallowed join strategies
    for join_strategy in ("cross", "anti", "semi"):
        with pytest.raises(
            ValueError,
            match=f"`how` must be one of {{'left', 'inner', 'full'}}; found '{join_strategy}'",
        ):
            a.update(b, how=join_strategy)  # type: ignore[arg-type]


def test_join_frame_consistency() -> None:
    df = pl.DataFrame({"A": [1, 2, 3]})
    ldf = pl.DataFrame({"A": [1, 2, 5]}).lazy()

    with pytest.raises(TypeError, match="expected `other`.*LazyFrame"):
        _ = ldf.join(df, on="A")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected `other`.*DataFrame"):
        _ = df.join(ldf, on="A")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected `other`.*LazyFrame"):
        _ = ldf.join_asof(df, on="A")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected `other`.*DataFrame"):
        _ = df.join_asof(ldf, on="A")  # type: ignore[arg-type]


def test_join_concat_projection_pd_case_7071() -> None:
    ldf = pl.DataFrame({"id": [1, 2], "value": [100, 200]}).lazy()
    ldf2 = pl.DataFrame({"id": [1, 3], "value": [100, 300]}).lazy()

    ldf = ldf.join(ldf2, on=["id", "value"])
    ldf = pl.concat([ldf, ldf2])
    result = ldf.select("id")

    expected = pl.DataFrame({"id": [1, 1, 3]}).lazy()
    assert_frame_equal(result, expected)


@pytest.mark.may_fail_auto_streaming  # legacy full join is not order-preserving whereas new-streaming is
def test_join_sorted_fast_paths_null() -> None:
    df1 = pl.DataFrame({"x": [0, 1, 0]}).sort("x")
    df2 = pl.DataFrame({"x": [0, None], "y": [0, 1]})
    assert df1.join(df2, on="x", how="inner").to_dict(as_series=False) == {
        "x": [0, 0],
        "y": [0, 0],
    }
    assert df1.join(df2, on="x", how="left").to_dict(as_series=False) == {
        "x": [0, 0, 1],
        "y": [0, 0, None],
    }
    assert df1.join(df2, on="x", how="anti").to_dict(as_series=False) == {"x": [1]}
    assert df1.join(df2, on="x", how="semi").to_dict(as_series=False) == {"x": [0, 0]}
    assert df1.join(df2, on="x", how="full").to_dict(as_series=False) == {
        "x": [0, 0, 1, None],
        "x_right": [0, 0, None, None],
        "y": [0, 0, None, 1],
    }


def test_full_outer_join_list_() -> None:
    schema = {"id": pl.Int64, "vals": pl.List(pl.Float64)}
    join_schema = {**schema, **{k + "_right": t for (k, t) in schema.items()}}
    df1 = pl.DataFrame({"id": [1], "vals": [[]]}, schema=schema)  # type: ignore[arg-type]
    df2 = pl.DataFrame({"id": [2, 3], "vals": [[], [4]]}, schema=schema)  # type: ignore[arg-type]
    expected = pl.DataFrame(
        {
            "id": [None, None, 1],
            "vals": [None, None, []],
            "id_right": [2, 3, None],
            "vals_right": [[], [4.0], None],
        },
        schema=join_schema,  # type: ignore[arg-type]
    )
    out = df1.join(df2, on="id", how="full", maintain_order="right_left")
    assert_frame_equal(out, expected)


@pytest.mark.slow
def test_join_validation() -> None:
    def test_each_join_validation(
        unique: pl.DataFrame, duplicate: pl.DataFrame, on: str, how: JoinStrategy
    ) -> None:
        # one_to_many
        _one_to_many_success_inner = unique.join(
            duplicate, on=on, how=how, validate="1:m"
        )

        with pytest.raises(ComputeError):
            _one_to_many_fail_inner = duplicate.join(
                unique, on=on, how=how, validate="1:m"
            )

        # one to one
        with pytest.raises(ComputeError):
            _one_to_one_fail_1_inner = unique.join(
                duplicate, on=on, how=how, validate="1:1"
            )

        with pytest.raises(ComputeError):
            _one_to_one_fail_2_inner = duplicate.join(
                unique, on=on, how=how, validate="1:1"
            )

        # many to one
        with pytest.raises(ComputeError):
            _many_to_one_fail_inner = unique.join(
                duplicate, on=on, how=how, validate="m:1"
            )

        _many_to_one_success_inner = duplicate.join(
            unique, on=on, how=how, validate="m:1"
        )

        # many to many
        _many_to_many_success_1_inner = duplicate.join(
            unique, on=on, how=how, validate="m:m"
        )

        _many_to_many_success_2_inner = unique.join(
            duplicate, on=on, how=how, validate="m:m"
        )

    # test data
    short_unique = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "id_str": ["1", "2", "3", "4"],
            "name": ["hello", "world", "rust", "polars"],
        }
    )
    short_duplicate = pl.DataFrame(
        {"id": [1, 2, 3, 1], "id_str": ["1", "2", "3", "1"], "cnt": [2, 4, 6, 1]}
    )
    long_unique = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "id_str": ["1", "2", "3", "4", "5"],
            "name": ["hello", "world", "rust", "polars", "meow"],
        }
    )
    long_duplicate = pl.DataFrame(
        {
            "id": [1, 2, 3, 1, 5],
            "id_str": ["1", "2", "3", "1", "5"],
            "cnt": [2, 4, 6, 1, 8],
        }
    )

    join_strategies: list[JoinStrategy] = ["inner", "full", "left"]

    for join_col in ["id", "id_str"]:
        for how in join_strategies:
            # same size
            test_each_join_validation(long_unique, long_duplicate, join_col, how)

            # left longer
            test_each_join_validation(long_unique, short_duplicate, join_col, how)

            # right longer
            test_each_join_validation(short_unique, long_duplicate, join_col, how)


@typing.no_type_check
def test_join_validation_many_keys() -> None:
    # unique in both
    df1 = pl.DataFrame(
        {
            "val1": [11, 12, 13, 14],
            "val2": [1, 2, 3, 4],
        }
    )
    df2 = pl.DataFrame(
        {
            "val1": [11, 12, 13, 14],
            "val2": [1, 2, 3, 4],
        }
    )
    for join_type in ["inner", "left", "full"]:
        for val in ["m:m", "m:1", "1:1", "1:m"]:
            df1.join(df2, on=["val1", "val2"], how=join_type, validate=val)

    # many in lhs
    df1 = pl.DataFrame(
        {
            "val1": [11, 11, 12, 13, 14],
            "val2": [1, 1, 2, 3, 4],
        }
    )

    for join_type in ["inner", "left", "full"]:
        for val in ["1:1", "1:m"]:
            with pytest.raises(ComputeError):
                df1.join(df2, on=["val1", "val2"], how=join_type, validate=val)

    # many in rhs
    df1 = pl.DataFrame(
        {
            "val1": [11, 12, 13, 14],
            "val2": [1, 2, 3, 4],
        }
    )
    df2 = pl.DataFrame(
        {
            "val1": [11, 11, 12, 13, 14],
            "val2": [1, 1, 2, 3, 4],
        }
    )

    for join_type in ["inner", "left", "full"]:
        for val in ["m:1", "1:1"]:
            with pytest.raises(ComputeError):
                df1.join(df2, on=["val1", "val2"], how=join_type, validate=val)


def test_full_outer_join_bool() -> None:
    df1 = pl.DataFrame({"id": [True, False], "val": [1, 2]})
    df2 = pl.DataFrame({"id": [True, False], "val": [0, -1]})
    assert df1.join(df2, on="id", how="full", maintain_order="right").to_dict(
        as_series=False
    ) == {
        "id": [True, False],
        "val": [1, 2],
        "id_right": [True, False],
        "val_right": [0, -1],
    }


def test_full_outer_join_coalesce_different_names_13450() -> None:
    df1 = pl.DataFrame({"L1": ["a", "b", "c"], "L3": ["b", "c", "d"], "L2": [1, 2, 3]})
    df2 = pl.DataFrame({"L3": ["a", "c", "d"], "R2": [7, 8, 9]})

    expected = pl.DataFrame(
        {
            "L1": ["a", "c", "d", "b"],
            "L3": ["b", "d", None, "c"],
            "L2": [1, 3, None, 2],
            "R2": [7, 8, 9, None],
        }
    )

    out = df1.join(df2, left_on="L1", right_on="L3", how="full", coalesce=True)
    assert_frame_equal(out, expected, check_row_order=False)


# https://github.com/pola-rs/polars/issues/10663
def test_join_on_wildcard_error() -> None:
    df = pl.DataFrame({"x": [1]})
    df2 = pl.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(
        InvalidOperationError,
    ):
        df.join(df2, on=pl.all())


def test_join_on_nth_error() -> None:
    df = pl.DataFrame({"x": [1]})
    df2 = pl.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(
        InvalidOperationError,
    ):
        df.join(df2, on=pl.first())


def test_join_results_in_duplicate_names() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [1, 2, 3],
            "c_right": [1, 2, 3],
        }
    )

    def f(x: Any) -> Any:
        return x.join(x, on=["a", "b"], how="left")

    # Ensure it also contains the hint
    match_str = "(?s)column with name 'c_right' already exists.*You may want to try"

    # Ensure it fails immediately when resolving schema.
    with pytest.raises(DuplicateError, match=match_str):
        f(df.lazy()).collect_schema()

    with pytest.raises(DuplicateError, match=match_str):
        f(df.lazy()).collect()

    with pytest.raises(DuplicateError, match=match_str):
        f(df).collect()


def test_join_duplicate_suffixed_columns_from_join_key_column_21048() -> None:
    df = pl.DataFrame({"a": 1, "b": 1, "b_right": 1})

    def f(x: Any) -> Any:
        return x.join(x, on="a")

    # Ensure it also contains the hint
    match_str = "(?s)column with name 'b_right' already exists.*You may want to try"

    # Ensure it fails immediately when resolving schema.
    with pytest.raises(DuplicateError, match=match_str):
        f(df.lazy()).collect_schema()

    with pytest.raises(DuplicateError, match=match_str):
        f(df.lazy()).collect()

    with pytest.raises(DuplicateError, match=match_str):
        f(df)


def test_join_projection_invalid_name_contains_suffix_15243() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    df2 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()

    with pytest.raises(ColumnNotFoundError):
        (
            df1.join(df2, on="a")
            .select(pl.col("b").filter(pl.col("b") == pl.col("foo_right")))
            .collect()
        )


def test_join_list_non_numeric() -> None:
    assert (
        pl.DataFrame(
            {
                "lists": [
                    ["a", "b", "c"],
                    ["a", "c", "b"],
                    ["a", "c", "b"],
                    ["a", "c", "d"],
                ]
            }
        )
    ).group_by("lists", maintain_order=True).agg(pl.len().alias("count")).to_dict(
        as_series=False
    ) == {
        "lists": [["a", "b", "c"], ["a", "c", "b"], ["a", "c", "d"]],
        "count": [1, 2, 1],
    }


@pytest.mark.slow
def test_join_4_columns_with_validity() -> None:
    # join on 4 columns so we trigger combine validities
    # use 138 as that is 2 u64 and a remainder
    a = pl.DataFrame(
        {"a": [None if a % 6 == 0 else a for a in range(138)]}
    ).with_columns(
        b=pl.col("a"),
        c=pl.col("a"),
        d=pl.col("a"),
    )

    assert a.join(a, on=["a", "b", "c", "d"], how="inner", nulls_equal=True).shape == (
        644,
        4,
    )
    assert a.join(a, on=["a", "b", "c", "d"], how="inner", nulls_equal=False).shape == (
        115,
        4,
    )


@pytest.mark.release
def test_cross_join() -> None:
    # triggers > 100 rows implementation
    # https://github.com/pola-rs/polars/blob/5f5acb2a523ce01bc710768b396762b8e69a9e07/polars/polars-core/src/frame/cross_join.rs#L34
    df1 = pl.DataFrame({"col1": ["a"], "col2": ["d"]})
    df2 = pl.DataFrame({"frame2": pl.arange(0, 100, eager=True)})
    out = df2.join(df1, how="cross")
    df2 = pl.DataFrame({"frame2": pl.arange(0, 101, eager=True)})
    assert_frame_equal(df2.join(df1, how="cross").slice(0, 100), out)


@pytest.mark.release
def test_cross_join_slice_pushdown() -> None:
    # this will likely go out of memory if we did not pushdown the slice
    df = (
        pl.Series("x", pl.arange(0, 2**16 - 1, eager=True, dtype=pl.UInt16) % 2**15)
    ).to_frame()

    result = df.lazy().join(df.lazy(), how="cross", suffix="_").slice(-5, 10).collect()
    expected = pl.DataFrame(
        {
            "x": [32766, 32766, 32766, 32766, 32766],
            "x_": [32762, 32763, 32764, 32765, 32766],
        },
        schema={"x": pl.UInt16, "x_": pl.UInt16},
    )
    assert_frame_equal(result, expected)

    result = df.lazy().join(df.lazy(), how="cross", suffix="_").slice(2, 10).collect()
    expected = pl.DataFrame(
        {
            "x": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x_": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        schema={"x": pl.UInt16, "x_": pl.UInt16},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_join_coalesce(how: JoinStrategy) -> None:
    a = pl.LazyFrame({"a": [1, 2], "b": [1, 2]})
    b = pl.LazyFrame(
        {
            "a": [1, 2, 1, 2],
            "b": [5, 7, 8, 9],
            "c": [1, 2, 1, 2],
        }
    )

    how = "inner"
    q = a.join(b, on="a", coalesce=False, how=how)
    out = q.collect()
    assert q.collect_schema() == out.schema
    assert out.columns == ["a", "b", "a_right", "b_right", "c"]

    q = a.join(b, on=["a", "b"], coalesce=False, how=how)
    out = q.collect()
    assert q.collect_schema() == out.schema
    assert out.columns == ["a", "b", "a_right", "b_right", "c"]

    q = a.join(b, on=["a", "b"], coalesce=True, how=how)
    out = q.collect()
    assert q.collect_schema() == out.schema
    assert out.columns == ["a", "b", "c"]


@pytest.mark.parametrize("how", ["left", "inner", "full"])
def test_join_empties(how: JoinStrategy) -> None:
    df1 = pl.DataFrame({"col1": [], "col2": [], "col3": []})
    df2 = pl.DataFrame({"col2": [], "col4": [], "col5": []})

    df = df1.join(df2, on="col2", how=how)
    assert df.height == 0


def test_join_raise_on_redundant_keys() -> None:
    left = pl.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7]})
    right = pl.DataFrame({"a": [2, 3, 4], "c": [4, 5, 6]})
    with pytest.raises(InvalidOperationError, match="already joined on"):
        left.join(right, on=["a", "a"], how="full", coalesce=True)


@pytest.mark.parametrize("coalesce", [False, True])
def test_join_raise_on_repeated_expression_key_names(coalesce: bool) -> None:
    left = pl.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7]})
    right = pl.DataFrame({"a": [2, 3, 4], "c": [4, 5, 6]})
    with (  # noqa: PT012
        pytest.raises(InvalidOperationError, match="already joined on"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter(action="ignore", category=UserWarning)
        left.join(
            right, on=[pl.col("a"), pl.col("a") % 2], how="full", coalesce=coalesce
        )


def test_join_lit_panic_11410() -> None:
    df = pl.LazyFrame({"date": [1, 2, 3], "symbol": [4, 5, 6]})
    dates = df.select("date").unique(maintain_order=True)
    symbols = df.select("symbol").unique(maintain_order=True)

    assert symbols.join(
        dates, left_on=pl.lit(1), right_on=pl.lit(1), maintain_order="left_right"
    ).collect().to_dict(as_series=False) == {
        "symbol": [4, 4, 4, 5, 5, 5, 6, 6, 6],
        "date": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }


def test_join_empty_literal_17027() -> None:
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame(schema={"a": pl.Int64})

    assert df1.join(df2, on=pl.lit(0), how="left").height == 1
    assert df1.join(df2, on=pl.lit(0), how="inner").height == 0
    assert (
        df1.lazy()
        .join(df2.lazy(), on=pl.lit(0), how="inner")
        .collect(engine="streaming")
        .height
        == 0
    )
    assert (
        df1.lazy()
        .join(df2.lazy(), on=pl.lit(0), how="left")
        .collect(engine="streaming")
        .height
        == 1
    )


@pytest.mark.parametrize(
    ("left_on", "right_on"),
    zip(
        [pl.col("a"), pl.col("a").sort(), [pl.col("a"), pl.col("b")]],
        [pl.col("a").slice(0, 2) * 2, pl.col("b"), [pl.col("a"), pl.col("b").head()]],
    ),
)
def test_join_non_elementwise_keys_raises(left_on: pl.Expr, right_on: pl.Expr) -> None:
    # https://github.com/pola-rs/polars/issues/17184
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    right = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = left.join(
        right,
        left_on=left_on,
        right_on=right_on,
        how="inner",
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect()


def test_join_coalesce_not_supported_warning() -> None:
    # https://github.com/pola-rs/polars/issues/17184
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    right = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

    q = left.join(
        right,
        left_on=[pl.col("a") * 2],
        right_on=[pl.col("a") * 2],
        how="inner",
        coalesce=True,
    )
    with pytest.warns(UserWarning, match="turning off key coalescing"):
        got = q.collect()
    expect = pl.DataFrame(
        {"a": [1, 2, 3], "b": [3, 4, 5], "a_right": [1, 2, 3], "b_right": [3, 4, 5]}
    )

    assert_frame_equal(expect, got, check_row_order=False)


@pytest.mark.parametrize(
    ("on_args"),
    [
        {"on": "a", "left_on": "a"},
        {"on": "a", "right_on": "a"},
        {"on": "a", "left_on": "a", "right_on": "a"},
    ],
)
def test_join_on_and_left_right_on(on_args: dict[str, str]) -> None:
    df1 = pl.DataFrame({"a": [1], "b": [2]})
    df2 = pl.DataFrame({"a": [1], "c": [3]})
    msg = "cannot use 'on' in conjunction with 'left_on' or 'right_on'"
    with pytest.raises(ValueError, match=msg):
        df1.join(df2, **on_args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("on_args"),
    [
        {"left_on": "a"},
        {"right_on": "a"},
    ],
)
def test_join_only_left_or_right_on(on_args: dict[str, str]) -> None:
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [1]})
    msg = "'left_on' requires corresponding 'right_on'"
    with pytest.raises(ValueError, match=msg):
        df1.join(df2, **on_args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("on_args"),
    [
        {"on": "a"},
        {"left_on": "a", "right_on": "a"},
    ],
)
def test_cross_join_no_on_keys(on_args: dict[str, str]) -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})
    msg = "cross join should not pass join keys"
    with pytest.raises(ValueError, match=msg):
        df1.join(df2, how="cross", **on_args)  # type: ignore[arg-type]


@pytest.mark.parametrize("set_sorted", [True, False])
def test_left_join_slice_pushdown_19405(set_sorted: bool) -> None:
    left = pl.LazyFrame({"k": [1, 2, 3, 4, 0]})
    right = pl.LazyFrame({"k": [1, 1, 1, 1, 0]})

    if set_sorted:
        # The data isn't actually sorted on purpose to ensure we default to a
        # hash join unless we set the sorted flag here, in case there is new
        # code in the future that automatically identifies sortedness during
        # Series construction from Python.
        left = left.set_sorted("k")
        right = right.set_sorted("k")

    q = left.join(right, on="k", how="left", maintain_order="left_right").head(5)
    assert_frame_equal(q.collect(), pl.DataFrame({"k": [1, 1, 1, 1, 2]}))


def test_join_key_type_coercion_19597() -> None:
    left = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Float64)})
    right = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})

    with pytest.raises(SchemaError, match="datatypes of join keys don't match"):
        left.join(right, left_on=pl.col("a"), right_on=pl.col("a")).collect_schema()

    with pytest.raises(SchemaError, match="datatypes of join keys don't match"):
        left.join(
            right, left_on=pl.col("a") * 2, right_on=pl.col("a") * 2
        ).collect_schema()


def test_array_explode_join_19763() -> None:
    q = pl.LazyFrame().select(
        pl.lit(pl.Series([[1], [2]], dtype=pl.Array(pl.Int64, 1))).explode().alias("k")
    )

    q = q.join(pl.LazyFrame({"k": [1, 2]}), on="k")

    assert_frame_equal(q.collect().sort("k"), pl.DataFrame({"k": [1, 2]}))


def test_join_full_19814() -> None:
    schema = {"a": pl.Int64, "c": pl.Categorical}
    a = pl.LazyFrame({"a": [1], "c": [None]}, schema=schema)
    b = pl.LazyFrame({"a": [1, 3, 4]})
    assert_frame_equal(
        a.join(b, on="a", how="full", coalesce=True).collect(),
        pl.DataFrame({"a": [1, 3, 4], "c": [None, None, None]}, schema=schema),
        check_row_order=False,
    )


def test_join_preserve_order_inner() -> None:
    left = pl.LazyFrame({"a": [None, 2, 1, 1, 5]})
    right = pl.LazyFrame({"a": [1, 1, None, 2], "b": [6, 7, 8, 9]})

    # Inner joins

    inner_left = left.join(right, on="a", how="inner", maintain_order="left").collect()
    assert inner_left.get_column("a").cast(pl.UInt32).to_list() == [2, 1, 1, 1, 1]
    inner_left_right = left.join(
        right, on="a", how="inner", maintain_order="left"
    ).collect()
    assert inner_left.get_column("a").equals(inner_left_right.get_column("a"))

    inner_right = left.join(
        right, on="a", how="inner", maintain_order="right"
    ).collect()
    assert inner_right.get_column("a").cast(pl.UInt32).to_list() == [1, 1, 1, 1, 2]
    inner_right_left = left.join(
        right, on="a", how="inner", maintain_order="right"
    ).collect()
    assert inner_right.get_column("a").equals(inner_right_left.get_column("a"))


# The new streaming engine does not provide the same maintain_order="none"
# ordering guarantee that is currently kept for compatibility on the in-memory
# engine.
@pytest.mark.may_fail_auto_streaming
def test_join_preserve_order_left() -> None:
    left = pl.LazyFrame({"a": [None, 2, 1, 1, 5]})
    right = pl.LazyFrame({"a": [1, None, 2, 6], "b": [6, 7, 8, 9]})

    # Right now the left join algorithm is ordered without explicitly setting any order
    # This behaviour is deprecated but can only be removed in 2.0
    left_none = left.join(right, on="a", how="left", maintain_order="none").collect()
    assert left_none.get_column("a").cast(pl.UInt32).to_list() == [
        None,
        2,
        1,
        1,
        5,
    ]

    left_left = left.join(right, on="a", how="left", maintain_order="left").collect()
    assert left_left.get_column("a").cast(pl.UInt32).to_list() == [
        None,
        2,
        1,
        1,
        5,
    ]

    left_left_right = left.join(
        right, on="a", how="left", maintain_order="left_right"
    ).collect()
    # If the left order is preserved then there are no unsorted right rows
    assert left_left.get_column("a").equals(left_left_right.get_column("a"))

    left_right = left.join(right, on="a", how="left", maintain_order="right").collect()
    assert left_right.get_column("a").cast(pl.UInt32).to_list()[:5] == [
        1,
        1,
        2,
        None,
        5,
    ]

    left_right_left = left.join(
        right, on="a", how="left", maintain_order="right_left"
    ).collect()
    assert left_right_left.get_column("a").cast(pl.UInt32).to_list() == [
        1,
        1,
        2,
        None,
        5,
    ]

    right_left = left.join(right, on="a", how="right", maintain_order="left").collect()
    assert right_left.get_column("a").cast(pl.UInt32).to_list() == [2, 1, 1, None, 6]

    right_right = left.join(
        right, on="a", how="right", maintain_order="right"
    ).collect()
    assert right_right.get_column("a").cast(pl.UInt32).to_list() == [
        1,
        1,
        None,
        2,
        6,
    ]


def test_join_preserve_order_full() -> None:
    left = pl.LazyFrame({"a": [None, 2, 1, 1, 5]})
    right = pl.LazyFrame({"a": [1, None, 2, 6], "b": [6, 7, 8, 9]})

    full_left = left.join(right, on="a", how="full", maintain_order="left").collect()
    assert full_left.get_column("a").cast(pl.UInt32).to_list()[:5] == [
        None,
        2,
        1,
        1,
        5,
    ]
    full_right = left.join(right, on="a", how="full", maintain_order="right").collect()
    assert full_right.get_column("a").cast(pl.UInt32).to_list()[:5] == [
        1,
        1,
        None,
        2,
        None,
    ]

    full_left_right = left.join(
        right, on="a", how="full", maintain_order="left_right"
    ).collect()
    assert full_left_right.get_column("a_right").cast(pl.UInt32).to_list() == [
        None,
        2,
        1,
        1,
        None,
        None,
        6,
    ]

    full_right_left = left.join(
        right, on="a", how="full", maintain_order="right_left"
    ).collect()
    assert full_right_left.get_column("a").cast(pl.UInt32).to_list() == [
        1,
        1,
        None,
        2,
        None,
        None,
        5,
    ]


@pytest.mark.parametrize(
    "dtypes",
    [
        ["Int128", "Int128", "Int64"],
        ["Int128", "Int128", "Int32"],
        ["Int128", "Int128", "Int16"],
        ["Int128", "Int128", "Int8"],
        ["Int128", "UInt64", "Int128"],
        ["Int128", "UInt64", "Int64"],
        ["Int128", "UInt64", "Int32"],
        ["Int128", "UInt64", "Int16"],
        ["Int128", "UInt64", "Int8"],
        ["Int128", "UInt32", "Int128"],
        ["Int128", "UInt16", "Int128"],
        ["Int128", "UInt8", "Int128"],

        ["Int64", "Int64", "Int32"],
        ["Int64", "Int64", "Int16"],
        ["Int64", "Int64", "Int8"],
        ["Int64", "UInt32", "Int64"],
        ["Int64", "UInt32", "Int32"],
        ["Int64", "UInt32", "Int16"],
        ["Int64", "UInt32", "Int8"],
        ["Int64", "UInt16", "Int64"],
        ["Int64", "UInt8", "Int64"],

        ["Int32", "Int32", "Int16"],
        ["Int32", "Int32", "Int8"],
        ["Int32", "UInt16", "Int32"],
        ["Int32", "UInt16", "Int16"],
        ["Int32", "UInt16", "Int8"],
        ["Int32", "UInt8", "Int32"],

        ["Int16", "Int16", "Int8"],
        ["Int16", "UInt8", "Int16"],
        ["Int16", "UInt8", "Int8"],

        ["UInt64", "UInt64", "UInt32"],
        ["UInt64", "UInt64", "UInt16"],
        ["UInt64", "UInt64", "UInt8"],

        ["UInt32", "UInt32", "UInt16"],
        ["UInt32", "UInt32", "UInt8"],

        ["UInt16", "UInt16", "UInt8"],

        ["Float64", "Float64", "Float32"],
    ],
)  # fmt: skip
@pytest.mark.parametrize("swap", [True, False])
def test_join_numeric_key_upcast_15338(
    dtypes: tuple[str, str, str], swap: bool
) -> None:
    supertype, ltype, rtype = (getattr(pl, x) for x in dtypes)
    ltype, rtype = (rtype, ltype) if swap else (ltype, rtype)

    left = pl.select(pl.Series("a", [1, 1, 3]).cast(ltype)).lazy()
    right = pl.select(pl.Series("a", [1]).cast(rtype), b=pl.lit("A")).lazy()

    assert_frame_equal(
        left.join(right, on="a", how="left").collect(),
        pl.select(a=pl.Series([1, 1, 3]).cast(ltype), b=pl.Series(["A", "A", None])),
        check_row_order=False,
    )

    assert_frame_equal(
        left.join(right, on="a", how="left", coalesce=False).drop("a_right").collect(),
        pl.select(a=pl.Series([1, 1, 3]).cast(ltype), b=pl.Series(["A", "A", None])),
        check_row_order=False,
    )

    assert_frame_equal(
        left.join(right, on="a", how="full").collect(),
        pl.select(
            a=pl.Series([1, 1, 3]).cast(ltype),
            a_right=pl.Series([1, 1, None]).cast(rtype),
            b=pl.Series(["A", "A", None]),
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        left.join(right, on="a", how="full", coalesce=True).collect(),
        pl.select(
            a=pl.Series([1, 1, 3]).cast(supertype),
            b=pl.Series(["A", "A", None]),
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        left.join(right, on="a", how="semi").collect(),
        pl.select(a=pl.Series([1, 1]).cast(ltype)),
    )

    # join_where
    for optimizations in [pl.QueryOptFlags(), pl.QueryOptFlags.none()]:
        assert_frame_equal(
            left.join_where(right, pl.col("a") == pl.col("a_right")).collect(
                optimizations=optimizations,
            ),
            pl.select(
                a=pl.Series([1, 1]).cast(ltype),
                a_right=pl.lit(1, dtype=rtype),
                b=pl.Series(["A", "A"]),
            ),
        )


def test_join_numeric_key_upcast_forbid_float_int() -> None:
    ltype = pl.Float64
    rtype = pl.Int128

    left = pl.LazyFrame({"a": [1.0, 0.0]}, schema={"a": ltype})
    right = pl.LazyFrame({"a": [1, 2]}, schema={"a": rtype})

    # Establish baseline: In a non-join context, comparisons between ltype and
    # rtype succeed even if the upcast is lossy.
    assert_frame_equal(
        left.with_columns(right.collect()["a"].alias("a_right"))
        .select(pl.col("a") == pl.col("a_right"))
        .collect(),
        pl.DataFrame({"a": [True, False]}),
    )

    with pytest.raises(SchemaError, match="datatypes of join keys don't match"):
        left.join(right, on="a", how="left").collect()

    for optimizations in [pl.QueryOptFlags(), pl.QueryOptFlags.none()]:
        with pytest.raises(
            SchemaError, match="'join_where' cannot compare Float64 with Int128"
        ):
            left.join_where(right, pl.col("a") == pl.col("a_right")).collect(
                optimizations=optimizations,
            )

        with pytest.raises(
            SchemaError, match="'join_where' cannot compare Float64 with Int128"
        ):
            left.join_where(
                right, pl.col("a") == (pl.col("a") == pl.col("a_right"))
            ).collect(optimizations=optimizations)


def test_join_numeric_key_upcast_order() -> None:
    # E.g. when we are joining on this expression:
    # * col('a') + 127
    #
    # and we want to upcast, ensure that we upcast like this:
    # * ( col('a') + 127 ) .cast(<type>)
    #
    # and *not* like this:
    # * ( col('a').cast(<type>) + lit(127).cast(<type>) )
    #
    # as otherwise the results would be different.

    left = pl.select(pl.Series("a", [1], dtype=pl.Int8)).lazy()
    right = pl.select(
        pl.Series("a", [1, 128, -128], dtype=pl.Int64), b=pl.lit("A")
    ).lazy()

    # col('a') in `left` is Int8, the result will overflow to become -128
    left_expr = pl.col("a") + 127

    assert_frame_equal(
        left.join(right, left_on=left_expr, right_on="a", how="inner").collect(),
        pl.DataFrame(
            {
                "a": pl.Series([1], dtype=pl.Int8),
                "a_right": pl.Series([-128], dtype=pl.Int64),
                "b": "A",
            }
        ),
    )

    assert_frame_equal(
        left.join_where(right, left_expr == pl.col("a_right")).collect(),
        pl.DataFrame(
            {
                "a": pl.Series([1], dtype=pl.Int8),
                "a_right": pl.Series([-128], dtype=pl.Int64),
                "b": "A",
            }
        ),
    )

    assert_frame_equal(
        (
            left.join(right, left_on=left_expr, right_on="a", how="full")
            .collect()
            .sort(pl.all())
        ),
        pl.DataFrame(
            {
                "a": pl.Series([1, None, None], dtype=pl.Int8),
                "a_right": pl.Series([-128, 1, 128], dtype=pl.Int64),
                "b": ["A", "A", "A"],
            }
        ).sort(pl.all()),
    )


def test_no_collapse_join_when_maintain_order_20725() -> None:
    df1 = pl.LazyFrame({"Fraction_1": [0, 25, 50, 75, 100]})
    df2 = pl.LazyFrame({"Fraction_2": [0, 1]})
    df3 = pl.LazyFrame({"Fraction_3": [0, 1]})

    ldf = df1.join(df2, how="cross", maintain_order="left_right").join(
        df3, how="cross", maintain_order="left_right"
    )

    df_pl_lazy = ldf.filter(pl.col("Fraction_1") == 100).collect()
    df_pl_eager = ldf.collect().filter(pl.col("Fraction_1") == 100)

    assert_frame_equal(df_pl_lazy, df_pl_eager)


def test_join_where_predicate_type_coercion_21009() -> None:
    left_frame = pl.LazyFrame(
        {
            "left_match": ["A", "B", "C", "D", "E", "F"],
            "left_date_start": range(6),
        }
    )

    right_frame = pl.LazyFrame(
        {
            "right_match": ["D", "E", "F", "G", "H", "I"],
            "right_date": range(6),
        }
    )

    # Note: Cannot eq the plans as the operand sides are non-deterministic

    q1 = left_frame.join_where(
        right_frame,
        pl.col("left_match") == pl.col("right_match"),
        pl.col("right_date") >= pl.col("left_date_start"),
    )

    plan = q1.explain().splitlines()
    assert plan[0].strip().startswith("FILTER")
    assert plan[1] == "FROM"
    assert plan[2].strip().startswith("INNER JOIN")

    q2 = left_frame.join_where(
        right_frame,
        pl.all_horizontal(pl.col("left_match") == pl.col("right_match")),
        pl.col("right_date") >= pl.col("left_date_start"),
    )

    plan = q2.explain().splitlines()
    assert plan[0].strip().startswith("FILTER")
    assert plan[1] == "FROM"
    assert plan[2].strip().startswith("INNER JOIN")

    assert_frame_equal(q1.collect(), q2.collect())


def test_join_right_predicate_pushdown_21142() -> None:
    left = pl.LazyFrame({"key": [1, 2, 4], "values": ["a", "b", "c"]})
    right = pl.LazyFrame({"key": [1, 2, 3], "values": ["d", "e", "f"]})

    rjoin = left.join(right, on="key", how="right")

    q = rjoin.filter(pl.col("values").is_null())

    expect = pl.select(
        pl.Series("values", [None], pl.String),
        pl.Series("key", [3], pl.Int64),
        pl.Series("values_right", ["f"], pl.String),
    )

    assert_frame_equal(q.collect(), expect)

    # Ensure for right join, filter on RHS key-columns are pushed down.
    q = rjoin.filter(pl.col("values_right").is_null())

    plan = q.explain()
    assert plan.index("FILTER") > plan.index("RIGHT PLAN ON")

    assert_frame_equal(q.collect(), expect.clear())


def test_join_where_nested_expr_21066() -> None:
    left = pl.LazyFrame({"a": [1, 2]})
    right = pl.LazyFrame({"a": [1]})

    q = left.join_where(right, pl.col("a") == (pl.col("a_right") + 1))

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 2, "a_right": 1}))


def test_select_after_join_where_20831() -> None:
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    )

    right = pl.LazyFrame(
        {
            "a": [1, 4, 3, 7, None, None, 1],
            "c": [2, 3, 4, 5, 6, 7, 8],
            "d": [6, None, 7, 8, -1, 2, 4],
        }
    )

    q = left.join_where(
        right, pl.col("b") * 2 <= pl.col("a_right"), pl.col("a") < pl.col("c_right")
    )

    assert_frame_equal(
        q.select("d").collect().sort("d"),
        pl.Series("d", [None, None, 7, 8, 8, 8]).to_frame(),
    )

    assert q.select(pl.len()).collect().item() == 6

    q = (
        left.join(right, how="cross")
        .filter(pl.col("b") * 2 <= pl.col("a_right"))
        .filter(pl.col("a") < pl.col("c_right"))
    )

    assert_frame_equal(
        q.select("d").collect().sort("d"),
        pl.Series("d", [None, None, 7, 8, 8, 8]).to_frame(),
    )

    assert q.select(pl.len()).collect().item() == 6


@pytest.mark.parametrize(
    ("dtype", "data"),
    [
        (pl.Struct, [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]),
        (pl.List, [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]),
        (pl.Array(pl.Int64, 2), [[1, 1], [2, 2], [3, 3], [4, 4]]),
    ],
)
def test_join_on_nested(dtype: PolarsDataType, data: list[Any]) -> None:
    lhs = pl.DataFrame(
        {
            "a": data[:3],
            "b": [1, 2, 3],
        }
    )
    rhs = pl.DataFrame(
        {
            "a": [data[3], data[1]],
            "c": [4, 2],
        }
    )

    assert_frame_equal(
        lhs.join(rhs, on="a", how="left", maintain_order="left"),
        pl.select(
            a=pl.Series(data[:3]),
            b=pl.Series([1, 2, 3]),
            c=pl.Series([None, 2, None]),
        ),
    )
    assert_frame_equal(
        lhs.join(rhs, on="a", how="right", maintain_order="right"),
        pl.select(
            b=pl.Series([None, 2]),
            a=pl.Series([data[3], data[1]]),
            c=pl.Series([4, 2]),
        ),
    )
    assert_frame_equal(
        lhs.join(rhs, on="a", how="inner"),
        pl.select(
            a=pl.Series([data[1]]),
            b=pl.Series([2]),
            c=pl.Series([2]),
        ),
    )
    assert_frame_equal(
        lhs.join(rhs, on="a", how="full", maintain_order="left_right"),
        pl.select(
            a=pl.Series(data[:3] + [None]),
            b=pl.Series([1, 2, 3, None]),
            a_right=pl.Series([None, data[1], None, data[3]]),
            c=pl.Series([None, 2, None, 4]),
        ),
    )
    assert_frame_equal(
        lhs.join(rhs, on="a", how="semi"),
        pl.select(
            a=pl.Series([data[1]]),
            b=pl.Series([2]),
        ),
    )
    assert_frame_equal(
        lhs.join(rhs, on="a", how="anti", maintain_order="left"),
        pl.select(
            a=pl.Series([data[0], data[2]]),
            b=pl.Series([1, 3]),
        ),
    )
    assert_frame_equal(
        lhs.join(rhs, how="cross", maintain_order="left_right"),
        pl.select(
            a=pl.Series([data[0], data[0], data[1], data[1], data[2], data[2]]),
            b=pl.Series([1, 1, 2, 2, 3, 3]),
            a_right=pl.Series([data[3], data[1], data[3], data[1], data[3], data[1]]),
            c=pl.Series([4, 2, 4, 2, 4, 2]),
        ),
    )


def test_empty_join_result_with_array_15474() -> None:
    lhs = pl.DataFrame(
        {
            "x": [1, 2],
            "y": pl.Series([[1, 2, 3], [4, 5, 6]], dtype=pl.Array(pl.Int64, 3)),
        }
    )
    rhs = pl.DataFrame({"x": [0]})
    result = lhs.join(rhs, on="x")
    expected = pl.DataFrame(schema={"x": pl.Int64, "y": pl.Array(pl.Int64, 3)})
    assert_frame_equal(result, expected)


@pytest.mark.slow
def test_join_where_eager_perf_21145() -> None:
    left = pl.Series("left", range(3_000)).to_frame()
    right = pl.Series("right", range(1_000)).to_frame()

    p = pl.col("left").is_between(pl.lit(0, dtype=pl.Int64), pl.col("right"))
    runtime_eager = time_func(lambda: left.join_where(right, p))
    runtime_lazy = time_func(lambda: left.lazy().join_where(right.lazy(), p).collect())
    runtime_ratio = runtime_eager / runtime_lazy

    # Pick as high as reasonably possible for CI stability
    # * Was observed to be >=5 seconds on the bugged version, so 3 is a safe bet.
    threshold = 3

    if runtime_ratio > threshold:
        msg = f"runtime_ratio ({runtime_ratio}) > {threshold}x ({runtime_eager = }, {runtime_lazy = })"
        raise ValueError(msg)


def test_select_len_after_semi_anti_join_21343() -> None:
    lhs = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rhs = pl.LazyFrame({"a": [1, 2, 3]})

    q = lhs.join(rhs, on="a", how="anti").select(pl.len())

    assert q.collect().item() == 0


def test_multi_leftjoin_empty_right_21701() -> None:
    parent_data = {
        "id": [1, 30, 80],
        "parent_field1": [3, 20, 17],
    }
    parent_df = pl.LazyFrame(parent_data)
    child_df = pl.LazyFrame(
        [],
        schema={"id": pl.Int32(), "parent_id": pl.Int32(), "child_field1": pl.Int32()},
    )
    subchild_df = pl.LazyFrame(
        [], schema={"child_id": pl.Int32(), "subchild_field1": pl.Int32()}
    )

    joined_df = parent_df.join(
        child_df.join(
            subchild_df, left_on=pl.col("id"), right_on=pl.col("child_id"), how="left"
        ),
        left_on=pl.col("id"),
        right_on=pl.col("parent_id"),
        how="left",
    )
    joined_df = joined_df.select("id", "parent_field1")
    assert_frame_equal(joined_df.collect(), parent_df.collect(), check_row_order=False)


@pytest.mark.parametrize("order", ["none", "left_right", "right_left"])
def test_join_null_equal(order: Literal["none", "left_right", "right_left"]) -> None:
    lhs = pl.DataFrame({"x": [1, None, None], "y": [1, 2, 3]})
    with_null = pl.DataFrame({"x": [1, None], "z": [1, 2]})
    without_null = pl.DataFrame({"x": [1, 3], "z": [1, 3]})
    check_row_order = order != "none"

    # Inner join.
    assert_frame_equal(
        lhs.join(with_null, on="x", nulls_equal=True, maintain_order=order),
        pl.DataFrame({"x": [1, None, None], "y": [1, 2, 3], "z": [1, 2, 2]}),
        check_row_order=check_row_order,
    )
    assert_frame_equal(
        lhs.join(without_null, on="x", nulls_equal=True),
        pl.DataFrame({"x": [1], "y": [1], "z": [1]}),
    )

    # Left join.
    assert_frame_equal(
        lhs.join(with_null, on="x", how="left", nulls_equal=True, maintain_order=order),
        pl.DataFrame({"x": [1, None, None], "y": [1, 2, 3], "z": [1, 2, 2]}),
        check_row_order=check_row_order,
    )
    assert_frame_equal(
        lhs.join(
            without_null, on="x", how="left", nulls_equal=True, maintain_order=order
        ),
        pl.DataFrame({"x": [1, None, None], "y": [1, 2, 3], "z": [1, None, None]}),
        check_row_order=check_row_order,
    )

    # Full join.
    assert_frame_equal(
        lhs.join(
            with_null,
            on="x",
            how="full",
            nulls_equal=True,
            coalesce=True,
            maintain_order=order,
        ),
        pl.DataFrame({"x": [1, None, None], "y": [1, 2, 3], "z": [1, 2, 2]}),
        check_row_order=check_row_order,
    )
    if order == "left_right":
        expected = pl.DataFrame(
            {
                "x": [1, None, None, None],
                "x_right": [1, None, None, 3],
                "y": [1, 2, 3, None],
                "z": [1, None, None, 3],
            }
        )
    else:
        expected = pl.DataFrame(
            {
                "x": [1, None, None, None],
                "x_right": [1, 3, None, None],
                "y": [1, None, 2, 3],
                "z": [1, 3, None, None],
            }
        )
    assert_frame_equal(
        lhs.join(
            without_null, on="x", how="full", nulls_equal=True, maintain_order=order
        ),
        expected,
        check_row_order=check_row_order,
        check_column_order=False,
    )


def test_join_categorical_21815() -> None:
    left = pl.DataFrame({"x": ["a", "b", "c", "d"]}).with_columns(
        xc=pl.col.x.cast(pl.Categorical)
    )
    right = pl.DataFrame({"x": ["c", "d", "e", "f"]}).with_columns(
        xc=pl.col.x.cast(pl.Categorical)
    )

    # As key.
    cat_key = left.join(right, on="xc", how="full")

    # As payload.
    cat_payload = left.join(right, on="x", how="full")

    expected = pl.DataFrame(
        {
            "x": ["a", "b", "c", "d", None, None],
            "x_right": [None, None, "c", "d", "e", "f"],
        }
    ).with_columns(
        xc=pl.col.x.cast(pl.Categorical),
        xc_right=pl.col.x_right.cast(pl.Categorical),
    )

    assert_frame_equal(
        cat_key, expected, check_row_order=False, check_column_order=False
    )
    assert_frame_equal(
        cat_payload, expected, check_row_order=False, check_column_order=False
    )


def test_join_where_nested_boolean() -> None:
    df1 = pl.DataFrame({"a": [1, 9, 22], "b": [6, 4, 50]})
    df2 = pl.DataFrame({"c": [1]})

    predicate = (pl.col("a") < pl.col("b")).cast(pl.Int32) < pl.col("c")
    result = df1.join_where(df2, predicate)
    expected = pl.DataFrame(
        {
            "a": [9],
            "b": [4],
            "c": [1],
        }
    )
    assert_frame_equal(result, expected)


def test_join_where_dtype_upcast() -> None:
    df1 = pl.DataFrame(
        {
            "a": pl.Series([1, 9, 22], dtype=pl.Int8),
            "b": [6, 4, 50],
        }
    )
    df2 = pl.DataFrame({"c": [10]})

    predicate = (pl.col("a") + (pl.col("b") > 0)) < pl.col("c")
    result = df1.join_where(df2, predicate)
    expected = pl.DataFrame(
        {
            "a": pl.Series([1], dtype=pl.Int8),
            "b": [6],
            "c": [10],
        }
    )
    assert_frame_equal(result, expected)


def test_join_where_valid_dtype_upcast_same_side() -> None:
    # Unsafe comparisons are all contained entirely within one table (LHS)
    # Safe comparisons across both tables.
    df1 = pl.DataFrame(
        {
            "a": pl.Series([1, 9, 22], dtype=pl.Float32),
            "b": [6, 4, 50],
        }
    )
    df2 = pl.DataFrame({"c": [10, 1, 5]})

    predicate = ((pl.col("a") < pl.col("b")).cast(pl.Int32) + 3) < pl.col("c")
    result = df1.join_where(df2, predicate).sort("a", "b", "c")
    expected = pl.DataFrame(
        {
            "a": pl.Series([1, 1, 9, 9, 22, 22], dtype=pl.Float32),
            "b": [6, 6, 4, 4, 50, 50],
            "c": [5, 10, 5, 10, 5, 10],
        }
    )
    assert_frame_equal(result, expected)


def test_join_where_invalid_dtype_upcast_different_side() -> None:
    # Unsafe comparisons exist across tables.
    df1 = pl.DataFrame(
        {
            "a": pl.Series([1, 9, 22], dtype=pl.Float32),
            "b": pl.Series([6, 4, 50], dtype=pl.Float64),
        }
    )
    df2 = pl.DataFrame({"c": [10, 1, 5]})

    predicate = ((pl.col("a") >= pl.col("c")) + 3) < 4
    with pytest.raises(
        SchemaError, match="'join_where' cannot compare Float32 with Int64"
    ):
        df1.join_where(df2, predicate)

    # add in a cast to predicate to fix
    predicate = ((pl.col("a").cast(pl.UInt8) >= pl.col("c")) + 3) < 4
    result = df1.join_where(df2, predicate).sort("a", "b", "c")
    expected = pl.DataFrame(
        {
            "a": pl.Series([1, 1, 9], dtype=pl.Float32),
            "b": pl.Series([6, 6, 4], dtype=pl.Float64),
            "c": [5, 10, 10],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Float32])
def test_join_where_literals(dtype: PolarsDataType) -> None:
    df1 = pl.DataFrame({"a": pl.Series([0, 1], dtype=dtype)})
    df2 = pl.DataFrame({"b": pl.Series([1, 2], dtype=dtype)})
    result = df1.join_where(df2, (pl.col("a") + pl.col("b")) < 2)
    expected = pl.DataFrame(
        {
            "a": pl.Series([0], dtype=dtype),
            "b": pl.Series([1], dtype=dtype),
        }
    )
    assert_frame_equal(result, expected)


def test_join_where_categorical_string_compare() -> None:
    dt = pl.Enum(["a", "b", "c"])
    df1 = pl.DataFrame({"a": pl.Series(["a", "a", "b", "c"], dtype=dt)})
    df2 = pl.DataFrame({"b": [1, 6, 4]})
    predicate = pl.col("a").is_in(["a", "b"]) & (pl.col("b") < 5)
    result = df1.join_where(df2, predicate).sort("a", "b")
    expected = pl.DataFrame(
        {
            "a": pl.Series(["a", "a", "a", "a", "b", "b"], dtype=dt),
            "b": [1, 1, 4, 4, 1, 4],
        }
    )
    assert_frame_equal(result, expected)


def test_join_where_nonboolean_predicate() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(
        ComputeError, match="'join_where' predicates must resolve to boolean"
    ):
        df1.join_where(df2, pl.col("a") * 2)


def test_empty_outer_join_22206() -> None:
    df = pl.LazyFrame({"a": [5, 6], "b": [1, 2]})
    empty = pl.LazyFrame(schema=df.collect_schema())
    assert_frame_equal(
        df.join(empty, on=["a", "b"], how="full", coalesce=True),
        df,
        check_row_order=False,
    )
    assert_frame_equal(
        empty.join(df, on=["a", "b"], how="full", coalesce=True),
        df,
        check_row_order=False,
    )


def test_join_coalesce_22498() -> None:
    df_a = pl.DataFrame({"y": [2]})
    df_b = pl.DataFrame({"x": [1], "y": [2]})
    df_j = df_a.lazy().join(df_b.lazy(), how="full", on="y", coalesce=True)
    assert_frame_equal(df_j.collect(), pl.DataFrame({"y": [2], "x": [1]}))


def _extract_plan_joins_and_filters(plan: str) -> list[str]:
    return [
        x
        for x in (x.strip() for x in plan.splitlines())
        if x.startswith("LEFT PLAN")  # noqa: PIE810
        or x.startswith("RIGHT PLAN")
        or x.startswith("FILTER")
    ]


def test_join_filter_pushdown_inner_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    # Filter on key output column is pushed to both sides.
    q = lhs.join(rhs, on=["a", "b"], how="inner", maintain_order="left_right").filter(
        pl.col("b") <= 2
    )

    expect = pl.DataFrame(
        {"a": [1, 2], "b": [1, 2], "c": ["a", "b"], "c_right": ["A", "B"]}
    )

    plan = q.explain()

    assert _extract_plan_joins_and_filters(plan) == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) <= (2)]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Side-specific filters are all pushed for inner join.
    q = (
        lhs.join(rhs, on=["a", "b"], how="inner", maintain_order="left_right")
        .filter(pl.col("b") <= 2)
        .filter(pl.col("c") == "a", pl.col("c_right") == "A")
    )

    expect = pl.DataFrame({"a": [1], "b": [1], "c": ["a"], "c_right": ["A"]})

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)

    assert extract[0] == 'LEFT PLAN ON: [col("a"), col("b")]'
    assert 'col("c")) == ("a")' in extract[1]
    assert 'col("b")) <= (2)' in extract[1]

    assert extract[2] == 'RIGHT PLAN ON: [col("a"), col("b")]'
    assert 'col("b")) <= (2)' in extract[3]
    assert 'col("c")) == ("A")' in extract[3]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filter applied to the non-coalesced `_right` column of an inner-join is
    # also pushed to the left
    # input table.
    q = lhs.join(
        rhs, on=["a", "b"], how="inner", coalesce=False, maintain_order="left_right"
    ).filter(pl.col("a_right") <= 2)

    expect = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
            "c": ["a", "b"],
            "a_right": [1, 2],
            "b_right": [1, 2],
            "c_right": ["A", "B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("a")) <= (2)]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("a")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Different names in left_on and right_on
    q = lhs.join(
        rhs, left_on="a", right_on="b", how="inner", maintain_order="left_right"
    ).filter(pl.col("a") <= 2)

    expect = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
            "c": ["a", "b"],
            "a_right": [1, 2],
            "c_right": ["A", "B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) <= (2)]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Different names in left_on and right_on, coalesce=False
    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="inner",
        coalesce=False,
        maintain_order="left_right",
    ).filter(pl.col("a") <= 2)

    expect = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
            "c": ["a", "b"],
            "a_right": [1, 2],
            "b_right": [1, 2],
            "c_right": ["A", "B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) <= (2)]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # left_on=col(A), right_on=lit(1). Filters referencing col(A) can only push
    # to the left side.
    q = lhs.join(
        rhs,
        left_on=["a", pl.lit(1)],
        right_on=[pl.lit(1), "b"],
        how="inner",
        coalesce=False,
        maintain_order="left_right",
    ).filter(
        pl.col("a") == 1,
        pl.col("b") >= 1,
        pl.col("a_right") <= 1,
        pl.col("b_right") >= 0,
    )

    expect = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": ["a"],
            "a_right": [1],
            "b_right": [1],
            "c_right": ["A"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)

    assert (
        extract[0]
        == 'LEFT PLAN ON: [col("a").cast(Int64), col("_POLARS_0").cast(Int64)]'
    )
    assert '(col("a")) == (1)' in extract[1]
    assert '(col("b")) >= (1)' in extract[1]
    assert (
        extract[2]
        == 'RIGHT PLAN ON: [col("_POLARS_1").cast(Int64), col("b").cast(Int64)]'
    )
    assert '(col("b")) >= (0)' in extract[3]
    assert 'col("a")) <= (1)' in extract[3]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters don't pass if they refer to columns from both tables
    # TODO: In the optimizer we can add additional equalities into the join
    # condition itself for some cases.
    q = lhs.join(rhs, on=["a"], how="inner", maintain_order="left_right").filter(
        pl.col("b") == pl.col("b_right")
    )

    expect = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 3],
            "c": ["a", "b", "c"],
            "b_right": [1, 2, 3],
            "c_right": ["A", "B", "C"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'FILTER [(col("b")) == (col("b_right"))]',
        'LEFT PLAN ON: [col("a")]',
        'RIGHT PLAN ON: [col("a")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Duplicate filter removal - https://github.com/pola-rs/polars/issues/23243
    q = (
        pl.LazyFrame({"x": [1, 2, 3]})
        .join(pl.LazyFrame({"x": [1, 2, 3]}), on="x", how="inner", coalesce=False)
        .filter(
            pl.col("x") == 2,
            pl.col("x_right") == 2,
        )
    )

    expect = pl.DataFrame(
        [
            pl.Series("x", [2], dtype=pl.Int64),
            pl.Series("x_right", [2], dtype=pl.Int64),
        ]
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("x")]',
        'FILTER [(col("x")) == (2)]',
        'RIGHT PLAN ON: [col("x")]',
        'FILTER [(col("x")) == (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_left_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    # Filter on key output column is pushed to both sides.
    q = lhs.join(rhs, on=["a", "b"], how="left", maintain_order="left_right").filter(
        pl.col("b") <= 2
    )

    expect = pl.DataFrame(
        {"a": [1, 2], "b": [1, 2], "c": ["a", "b"], "c_right": ["A", "B"]}
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) <= (2)]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filter on key output column is pushed to both sides.
    # This tests joins on differing left/right names.
    q = lhs.join(
        rhs, left_on="a", right_on="b", how="left", maintain_order="left_right"
    ).filter(pl.col("a") <= 2)

    expect = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
            "c": ["a", "b"],
            "a_right": [1, 2],
            "c_right": ["A", "B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) <= (2)]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters referring to columns that exist only in the left table can be pushed.
    q = lhs.join(rhs, on=["a", "b"], how="left", maintain_order="left_right").filter(
        pl.col("c") == "b"
    )

    expect = pl.DataFrame({"a": [2], "b": [2], "c": ["b"], "c_right": ["B"]})

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("c")) == ("b")]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters referring to columns that exist only in the right table cannot be
    # pushed for left-join
    q = lhs.join(rhs, on=["a", "b"], how="left", maintain_order="left_right").filter(
        # Note: `eq_missing` to block join downgrade.
        pl.col("c_right").eq_missing("B")
    )

    expect = pl.DataFrame({"a": [2], "b": [2], "c": ["b"], "c_right": ["B"]})

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'FILTER [(col("c_right")) ==v ("B")]',
        'LEFT PLAN ON: [col("a"), col("b")]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters referring to a non-coalesced key column originating from the right
    # table cannot be pushed.
    #
    # Note, technically it's possible to push these filters if we can guarantee that
    # they do not remove NULLs (or otherwise if we also apply the filter on the
    # result table). But this is not something we do at the moment.
    q = lhs.join(
        rhs, on=["a", "b"], how="left", coalesce=False, maintain_order="left_right"
    ).filter(pl.col("b_right").eq_missing(2))

    expect = pl.DataFrame(
        {
            "a": [2],
            "b": [2],
            "c": ["b"],
            "a_right": [2],
            "b_right": [2],
            "c_right": ["B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'FILTER [(col("b_right")) ==v (2)]',
        'LEFT PLAN ON: [col("a"), col("b")]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_right_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    # Filter on key output column is pushed to both sides.
    q = lhs.join(rhs, on=["a", "b"], how="right", maintain_order="left_right").filter(
        pl.col("b") <= 2
    )

    expect = pl.DataFrame(
        {"c": ["a", "b"], "a": [1, 2], "b": [1, 2], "c_right": ["A", "B"]}
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) <= (2)]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filter on key output column is pushed to both sides.
    # This tests joins on differing left/right names.
    # col(A) is coalesced into col(B) (from right), but col(B) is named as
    # col(B_right) in the output because the LHS table also has a col(B).
    q = lhs.join(
        rhs, left_on="a", right_on="b", how="right", maintain_order="left_right"
    ).filter(pl.col("b_right") <= 2)

    expect = pl.DataFrame(
        {
            "b": [1, 2],
            "c": ["a", "b"],
            "a": [1, 2],
            "b_right": [1, 2],
            "c_right": ["A", "B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) <= (2)]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("b")) <= (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters referring to columns that exist only in the right table can be pushed.
    q = lhs.join(rhs, on=["a", "b"], how="right", maintain_order="left_right").filter(
        pl.col("c_right") == "B"
    )

    expect = pl.DataFrame({"c": ["b"], "a": [2], "b": [2], "c_right": ["B"]})

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("c")) == ("B")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters referring to columns that exist only in the left table cannot be
    # pushed for right-join
    q = lhs.join(rhs, on=["a", "b"], how="right", maintain_order="left_right").filter(
        # Note: eq_missing to block join downgrade
        pl.col("c").eq_missing("b")
    )

    expect = pl.DataFrame({"c": ["b"], "a": [2], "b": [2], "c_right": ["B"]})

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'FILTER [(col("c")) ==v ("b")]',
        'LEFT PLAN ON: [col("a"), col("b")]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Filters referring to a non-coalesced key column originating from the left
    # table cannot be pushed for right-join.
    q = lhs.join(
        rhs, on=["a", "b"], how="right", coalesce=False, maintain_order="left_right"
    ).filter(pl.col("b").eq_missing(2))

    expect = pl.DataFrame(
        {
            "a": [2],
            "b": [2],
            "c": ["b"],
            "a_right": [2],
            "b_right": [2],
            "c_right": ["B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)
    assert extract == [
        'FILTER [(col("b")) ==v (2)]',
        'LEFT PLAN ON: [col("a"), col("b")]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_full_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    # Full join can only push filters that refer to coalesced key columns.
    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="full",
        coalesce=True,
        maintain_order="left_right",
    ).filter(pl.col("a") == 2)

    expect = pl.DataFrame(
        {
            "a": [2],
            "b": [2],
            "c": ["b"],
            "a_right": [2],
            "c_right": ["B"],
        }
    )

    plan = q.explain()
    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) == (2)]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("b")) == (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Non-coalescing full-join cannot push any filters
    # Note: We add fill_null to bypass non-NULL filter mask detection.
    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="full",
        coalesce=False,
        maintain_order="left_right",
    ).filter(
        pl.col("a").fill_null(0) >= 2,
        pl.col("a").fill_null(0) <= 2,
    )

    expect = pl.DataFrame(
        {
            "a": [2],
            "b": [2],
            "c": ["b"],
            "a_right": [2],
            "b_right": [2],
            "c_right": ["B"],
        }
    )

    plan = q.explain()
    extract = _extract_plan_joins_and_filters(plan)

    assert extract[0].startswith("FILTER ")
    assert extract[1:] == [
        'LEFT PLAN ON: [col("a")]',
        'RIGHT PLAN ON: [col("b")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_semi_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    q = lhs.join(
        rhs,
        left_on=["a", "b"],
        right_on=["b", pl.lit(2)],
        how="semi",
        maintain_order="left_right",
    ).filter(pl.col("a") == 2, pl.col("b") == 2, pl.col("c") == "b")

    expect = pl.DataFrame(
        {
            "a": [2],
            "b": [2],
            "c": ["b"],
        }
    )

    plan = q.explain()
    extract = _extract_plan_joins_and_filters(plan)

    # * filter on col(a) is pushed to both sides (renamed to col(b) in the right side)
    # * filter on col(b) is pushed only to left, as the right join key is a literal
    # * filter on col(c) is pushed only to left, as the column does not exist in
    #   the right.

    assert extract[0] == 'LEFT PLAN ON: [col("a"), col("b").cast(Int64)]'
    assert 'col("a")) == (2)' in extract[1]
    assert 'col("b")) == (2)' in extract[1]
    assert 'col("c")) == ("b")' in extract[1]

    assert extract[2:] == [
        'RIGHT PLAN ON: [col("b"), col("_POLARS_0").cast(Int64)]',
        'FILTER [(col("b")) == (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_anti_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    q = lhs.join(
        rhs,
        left_on=["a", "b"],
        right_on=["b", pl.lit(1)],
        how="anti",
        maintain_order="left_right",
    ).filter(pl.col("a") == 2, pl.col("b") == 2, pl.col("c") == "b")

    expect = pl.DataFrame(
        {
            "a": [2],
            "b": [2],
            "c": ["b"],
        }
    )

    plan = q.explain()
    extract = _extract_plan_joins_and_filters(plan)

    assert extract[0] == 'LEFT PLAN ON: [col("a"), col("b").cast(Int64)]'
    assert 'col("a")) == (2)' in extract[1]
    assert 'col("b")) == (2)' in extract[1]
    assert 'col("c")) == ("b")' in extract[1]

    assert extract[2:] == [
        'RIGHT PLAN ON: [col("b"), col("_POLARS_0").cast(Int64)]',
        'FILTER [(col("b")) == (2)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_cross_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [0, 0, 0, 0, 0], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    # Nested loop join for `!=`
    q = (
        lhs.with_row_index()
        .join(rhs, how="cross")
        .filter(
            pl.col("a") <= 4, pl.col("c_right") <= "B", pl.col("a") != pl.col("a_right")
        )
        .sort("index")
    )

    expect = pl.DataFrame(
        [
            pl.Series("index", [0, 0, 1, 1, 2, 2, 3, 3], dtype=pl.get_index_type()),
            pl.Series("a", [1, 1, 2, 2, 3, 3, 4, 4], dtype=pl.Int64),
            pl.Series("b", [1, 1, 2, 2, 3, 3, 4, 4], dtype=pl.Int64),
            pl.Series("c", ["a", "a", "b", "b", "c", "c", "d", "d"], dtype=pl.String),
            pl.Series("a_right", [0, 0, 0, 0, 0, 0, 0, 0], dtype=pl.Int64),
            pl.Series("b_right", [1, 2, 1, 2, 1, 2, 1, 2], dtype=pl.Int64),
            pl.Series(
                "c_right", ["A", "B", "A", "B", "A", "B", "A", "B"], dtype=pl.String
            ),
        ]
    )

    plan = q.explain()

    assert 'NESTED LOOP JOIN ON [(col("a")) != (col("a_right"))]' in plan

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        "LEFT PLAN:",
        'FILTER [(col("a")) <= (4)]',
        "RIGHT PLAN:",
        'FILTER [(col("c")) <= ("B")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Conversion to inner-join for `==`
    q = lhs.join(rhs, how="cross", maintain_order="left_right").filter(
        pl.col("a") <= 4,
        pl.col("c_right") <= "B",
        pl.col("a") == (pl.col("a_right") + 1),
    )

    expect = pl.DataFrame(
        {
            "a": [1, 1],
            "b": [1, 1],
            "c": ["a", "a"],
            "a_right": [0, 0],
            "b_right": [1, 2],
            "c_right": ["A", "B"],
        }
    )

    plan = q.explain()

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) <= (4)]',
        'RIGHT PLAN ON: [[(col("a")) + (1)]]',
        'FILTER [(col("c")) <= ("B")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_iejoin() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    q = (
        lhs.with_row_index()
        .join_where(
            rhs,
            pl.col("a") >= 1,
            pl.col("a") == pl.col("a_right"),
            pl.col("c_right") <= "B",
        )
        .sort("index")
    )

    expect = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 2],
            "c": ["a", "b"],
            "a_right": [1, 2],
            "b_right": [1, 2],
            "c_right": ["A", "B"],
        }
    ).with_row_index()

    plan = q.explain()

    assert "INNER JOIN" in plan

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) >= (1)]',
        'RIGHT PLAN ON: [col("a")]',
        'FILTER [(col("c")) <= ("B")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    q = (
        lhs.with_row_index()
        .join_where(
            rhs,
            pl.col("a") >= 1,
            pl.col("a") >= pl.col("a_right"),
            pl.col("c_right") <= "B",
        )
        .sort("index")
    )

    expect = pl.DataFrame(
        [
            pl.Series("index", [0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=pl.get_index_type()),
            pl.Series("a", [1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=pl.Int64),
            pl.Series("b", [1, 2, 2, 3, 3, 4, 4, None, None], dtype=pl.Int64),
            pl.Series(
                "c", ["a", "b", "b", "c", "c", "d", "d", "e", "e"], dtype=pl.String
            ),
            pl.Series("a_right", [1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=pl.Int64),
            pl.Series("b_right", [1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=pl.Int64),
            pl.Series(
                "c_right",
                ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
                dtype=pl.String,
            ),
        ]
    )

    plan = q.explain()

    assert "IEJOIN" in plan

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) >= (1)]',
        'RIGHT PLAN ON: [col("a")]',
        'FILTER [(col("c")) <= ("B")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_asof_join() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    q = lhs.join_asof(
        rhs,
        left_on=pl.col("a").set_sorted(),
        right_on=pl.col("b").set_sorted(),
        tolerance=0,
    ).filter(
        pl.col("a") >= 2,
        pl.col("b") >= 3,
        pl.col("c") >= "A",
        pl.col("c_right") >= "B",
    )

    expect = pl.DataFrame(
        {
            "a": [3],
            "b": [3],
            "c": ["c"],
            "a_right": [3],
            "b_right": [3],
            "c_right": ["C"],
        }
    )

    plan = q.explain()
    extract = _extract_plan_joins_and_filters(plan)

    assert extract[:2] == [
        'FILTER [(col("c_right")) >= ("B")]',
        'LEFT PLAN ON: [col("a").set_sorted()]',
    ]

    assert 'col("b")) >= (3)' in extract[2]
    assert 'col("c")) >= ("A")' in extract[2]
    assert 'col("a")) >= (2)' in extract[2]

    assert extract[3:] == ['RIGHT PLAN ON: [col("b").set_sorted()]']

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # With "by" columns
    q = lhs.join_asof(
        rhs,
        left_on="a",
        right_on="b",
        tolerance=99,
        by_left="b",
        by_right="a",
    ).filter(
        pl.col("a") >= 2,
        pl.col("b") >= 3,
        pl.col("c") >= "A",
        pl.col("c_right") >= "B",
    )

    expect = pl.DataFrame(
        {
            "a": [3],
            "b": [3],
            "c": ["c"],
            "b_right": [3],
            "c_right": ["C"],
        }
    )

    plan = q.explain()
    extract = _extract_plan_joins_and_filters(plan)

    assert extract[:2] == [
        'FILTER [(col("c_right")) >= ("B")]',
        'LEFT PLAN ON: [col("a")]',
    ]
    assert 'col("a")) >= (2)' in extract[2]
    assert 'col("b")) >= (3)' in extract[2]

    assert extract[3:] == [
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("a")) >= (3)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_full_join_rewrite() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, None],
            "b": [1, 2, 3, None, 5],
            "c": ["A", "B", "C", "D", "E"],
        }
    )

    # Downgrades to left-join
    q = lhs.join(rhs, on=["a", "b"], how="full", maintain_order="left_right").filter(
        pl.col("b") >= 3
    )

    expect = pl.DataFrame(
        [
            pl.Series("a", [3, 4], dtype=pl.Int64),
            pl.Series("b", [3, 4], dtype=pl.Int64),
            pl.Series("c", ["c", "d"], dtype=pl.String),
            pl.Series("a_right", [3, None], dtype=pl.Int64),
            pl.Series("b_right", [3, None], dtype=pl.Int64),
            pl.Series("c_right", ["C", None], dtype=pl.String),
        ]
    )

    plan = q.explain()

    assert "FULL JOIN" not in plan
    assert plan.startswith("LEFT JOIN")

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) >= (3)]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) >= (3)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Downgrades to right-join
    q = lhs.join(
        rhs, left_on="a", right_on="b", how="full", maintain_order="left_right"
    ).filter(pl.col("b_right") >= 3)

    expect = pl.DataFrame(
        [
            pl.Series("a", [3, 5], dtype=pl.Int64),
            pl.Series("b", [3, None], dtype=pl.Int64),
            pl.Series("c", ["c", "e"], dtype=pl.String),
            pl.Series("a_right", [3, None], dtype=pl.Int64),
            pl.Series("b_right", [3, 5], dtype=pl.Int64),
            pl.Series("c_right", ["C", "E"], dtype=pl.String),
        ]
    )

    plan = q.explain()

    assert "FULL JOIN" not in plan
    assert "RIGHT JOIN" in plan

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'FILTER [(col("a")) >= (3)]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("b")) >= (3)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Downgrades to right-join
    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="full",
        coalesce=True,
        maintain_order="left_right",
    ).filter(
        (pl.col("a") >= 1) | pl.col("a").is_null(),  # col(a) from LHS
        pl.col("a_right") >= 3,  # col(a) from RHS
        (pl.col("b") >= 2) | pl.col("b").is_null(),  # col(b) from LHS
        pl.col("c_right") >= "C",  # col(c) from RHS
    )

    expect = pl.DataFrame(
        [
            pl.Series("a", [3, None], dtype=pl.Int64),
            pl.Series("b", [3, None], dtype=pl.Int64),
            pl.Series("c", ["c", None], dtype=pl.String),
            pl.Series("a_right", [3, 4], dtype=pl.Int64),
            pl.Series("c_right", ["C", "D"], dtype=pl.String),
        ]
    )

    plan = q.explain()

    assert "FULL JOIN" not in plan
    assert "RIGHT JOIN" in plan

    extract = _extract_plan_joins_and_filters(plan)

    assert [
        'FILTER [([(col("b")) >= (2)]) | (col("b").is_null())]',
        'LEFT PLAN ON: [col("a")]',
        'FILTER [([(col("a")) >= (1)]) | (col("a").is_null())]',
        'RIGHT PLAN ON: [col("b")]',
    ]

    assert 'col("a")) >= (3)' in extract[4]
    assert '(col("b")) >= (1)]) | (col("b").alias("a").is_null())' in extract[4]
    assert 'col("c")) >= ("C")' in extract[4]

    assert len(extract) == 5

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Downgrades to inner-join
    q = lhs.join(rhs, on=["a", "b"], how="full", maintain_order="left_right").filter(
        pl.col("b").is_not_null(), pl.col("b_right").is_not_null()
    )

    expect = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], dtype=pl.Int64),
            pl.Series("b", [1, 2, 3], dtype=pl.Int64),
            pl.Series("c", ["a", "b", "c"], dtype=pl.String),
            pl.Series("a_right", [1, 2, 3], dtype=pl.Int64),
            pl.Series("b_right", [1, 2, 3], dtype=pl.Int64),
            pl.Series("c_right", ["A", "B", "C"], dtype=pl.String),
        ]
    )

    plan = q.explain()

    assert "FULL JOIN" not in plan
    assert plan.startswith("INNER JOIN")

    extract = _extract_plan_joins_and_filters(plan)

    assert extract[0] == 'LEFT PLAN ON: [col("a"), col("b")]'
    assert 'col("b").is_not_null()' in extract[1]
    assert 'col("b").alias("b_right").is_not_null()' in extract[1]

    assert extract[2] == 'RIGHT PLAN ON: [col("a"), col("b")]'
    assert 'col("b").is_not_null()' in extract[3]
    assert 'col("b").alias("b_right").is_not_null()' in extract[3]

    assert len(extract) == 4

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)

    # Does not downgrade because col(b) is a coalesced key-column, but the filter
    # is still pushed to both sides.
    q = lhs.join(
        rhs, on=["a", "b"], how="full", coalesce=True, maintain_order="left_right"
    ).filter(pl.col("b") >= 3)

    expect = pl.DataFrame(
        [
            pl.Series("a", [3, 4, None], dtype=pl.Int64),
            pl.Series("b", [3, 4, 5], dtype=pl.Int64),
            pl.Series("c", ["c", "d", None], dtype=pl.String),
            pl.Series("c_right", ["C", None, "E"], dtype=pl.String),
        ]
    )

    plan = q.explain()
    assert plan.startswith("FULL JOIN")

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) >= (3)]',
        'RIGHT PLAN ON: [col("a"), col("b")]',
        'FILTER [(col("b")) >= (3)]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_right_join_rewrite() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    # Downgrades to inner-join
    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="right",
        coalesce=True,
        maintain_order="left_right",
    ).filter(
        pl.col("a") <= 7,  # col(a) from RHS (LHS col(a) is coalesced into col(b_right))
        pl.col("b_right") <= 10,  # Key-column filter
        pl.col("c") <= "b",  # col(c) from LHS
    )

    expect = pl.DataFrame(
        [
            pl.Series("b", [1, 2], dtype=pl.Int64),
            pl.Series("c", ["a", "b"], dtype=pl.String),
            pl.Series("a", [1, 2], dtype=pl.Int64),
            pl.Series("b_right", [1, 2], dtype=pl.Int64),
            pl.Series("c_right", ["A", "B"], dtype=pl.String),
        ]
    )

    plan = q.explain()

    assert "RIGHT JOIN" not in plan
    assert "INNER JOIN" in plan

    extract = _extract_plan_joins_and_filters(plan)

    assert extract[0] == 'LEFT PLAN ON: [col("a")]'
    assert 'col("a")) <= (10)' in extract[1]
    assert 'col("c")) <= ("b")' in extract[1]

    assert extract[2] == 'RIGHT PLAN ON: [col("b")]'
    assert 'col("a")) <= (7)' in extract[3]
    assert 'col("b")) <= (10)' in extract[3]

    assert len(extract) == 4

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_join_rewrite_equality_above_and() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", "C", "D", "E"]}
    )

    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="full",
        coalesce=False,
        maintain_order="left_right",
    ).filter(((pl.col("b") >= 3) & False) >= False)

    expect = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3, 4, 5, None], dtype=pl.Int64),
            pl.Series("b", [1, 2, 3, 4, None, None], dtype=pl.Int64),
            pl.Series("c", ["a", "b", "c", "d", "e", None], dtype=pl.String),
            pl.Series("a_right", [1, 2, 3, None, 5, 4], dtype=pl.Int64),
            pl.Series("b_right", [1, 2, 3, None, 5, None], dtype=pl.Int64),
            pl.Series("c_right", ["A", "B", "C", None, "E", "D"], dtype=pl.String),
        ]
    )

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_left_join_rewrite() -> None:
    lhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, None], "c": ["a", "b", "c", "d", "e"]}
    )
    rhs = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, None, 5], "c": ["A", "B", None, "D", "E"]}
    )

    # Downgrades to inner-join
    q = lhs.join(
        rhs,
        left_on="a",
        right_on="b",
        how="left",
        coalesce=True,
        maintain_order="left_right",
    ).filter(pl.col("c_right") <= "B")

    expect = pl.DataFrame(
        [
            pl.Series("a", [1, 2], dtype=pl.Int64),
            pl.Series("b", [1, 2], dtype=pl.Int64),
            pl.Series("c", ["a", "b"], dtype=pl.String),
            pl.Series("a_right", [1, 2], dtype=pl.Int64),
            pl.Series("c_right", ["A", "B"], dtype=pl.String),
        ]
    )

    plan = q.explain()

    assert "LEFT JOIN" not in plan
    assert plan.startswith("INNER JOIN")

    extract = _extract_plan_joins_and_filters(plan)

    assert extract == [
        'LEFT PLAN ON: [col("a")]',
        'RIGHT PLAN ON: [col("b")]',
        'FILTER [(col("c")) <= ("B")]',
    ]

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_filter_pushdown_left_join_rewrite_23133() -> None:
    lhs = pl.LazyFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )

    rhs = pl.LazyFrame(
        {
            "apple": ["x", "y", "z"],
            "ham": ["a", "b", "d"],
            "bar": ["a", "b", "c"],
            "foo2": [1, 2, 3],
        }
    )

    q = lhs.join(rhs, how="left", on="ham", maintain_order="left_right").filter(
        pl.col("ham") == "a", pl.col("apple") == "x", pl.col("foo") <= 2
    )

    expect = pl.DataFrame(
        [
            pl.Series("foo", [1], dtype=pl.Int64),
            pl.Series("bar", [6.0], dtype=pl.Float64),
            pl.Series("ham", ["a"], dtype=pl.String),
            pl.Series("apple", ["x"], dtype=pl.String),
            pl.Series("bar_right", ["a"], dtype=pl.String),
            pl.Series("foo2", [1], dtype=pl.Int64),
        ]
    )

    plan = q.explain()
    assert "FULL JOIN" not in plan
    assert plan.startswith("INNER JOIN")

    extract = _extract_plan_joins_and_filters(plan)

    assert extract[0] == 'LEFT PLAN ON: [col("ham")]'
    assert '(col("foo")) <= (2)' in extract[1]
    assert 'col("ham")) == ("a")' in extract[1]

    assert extract[2] == 'RIGHT PLAN ON: [col("ham")]'
    assert 'col("ham")) == ("a")' in extract[3]
    assert 'col("apple")) == ("x")' in extract[3]

    assert len(extract) == 4

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


def test_join_rewrite_panic_23307() -> None:
    lhs = pl.select(a=pl.lit(1, dtype=pl.Int8)).lazy()
    rhs = pl.select(a=pl.lit(1, dtype=pl.Int16), x=pl.lit(1, dtype=pl.Int32)).lazy()

    q = lhs.join(rhs, on="a", how="left", coalesce=True).filter(pl.col("x") >= 1)

    assert_frame_equal(
        q.collect(),
        pl.select(
            a=pl.lit(1, dtype=pl.Int8),
            x=pl.lit(1, dtype=pl.Int32),
        ),
    )

    lhs = pl.select(a=pl.lit(999, dtype=pl.Int16)).lazy()

    # Note: -25 matches to (999).overflowing_cast(Int8).
    # This is specially chosen to test that we don't accidentally push the filter
    # to the RHS.
    rhs = pl.LazyFrame(
        {"a": [1, -25], "x": [1, 2]}, schema={"a": pl.Int8, "x": pl.Int32}
    )

    q = lhs.join(
        rhs,
        on=pl.col("a").cast(pl.Int8, strict=False, wrap_numerical=True),
        how="left",
        coalesce=False,
    ).filter(pl.col("a") >= 0)

    expect = pl.DataFrame(
        {"a": 999, "a_right": -25, "x": 2},
        schema={"a": pl.Int16, "a_right": pl.Int8, "x": pl.Int32},
    )

    plan = q.explain()

    assert not plan.startswith("FILTER")

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)


@pytest.mark.parametrize(
    ("expr_first_input", "expr_func"),
    [
        (pl.lit(None, dtype=pl.Int64), lambda col: col >= 1),
        (pl.lit(None, dtype=pl.Int64), lambda col: (col >= 1).is_not_null()),
        (pl.lit(None, dtype=pl.Int64), lambda col: (~(col >= 1)).is_not_null()),
        (pl.lit(None, dtype=pl.Int64), lambda col: ~(col >= 1).is_null()),
        #
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_in([1])),
        (pl.lit(None, dtype=pl.Int64), lambda col: ~col.is_in([1])),
        #
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_between(1, 1)),
        (1, lambda col: col.is_between(None, 1)),
        (1, lambda col: col.is_between(1, None)),
        #
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_close(1)),
        (1, lambda col: col.is_close(pl.lit(None, dtype=pl.Int64))),
        #
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_nan()),
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_not_nan()),
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_finite()),
        (pl.lit(None, dtype=pl.Int64), lambda col: col.is_infinite()),
        #
        (pl.lit(None, dtype=pl.Float64), lambda col: col.is_nan()),
        (pl.lit(None, dtype=pl.Float64), lambda col: col.is_not_nan()),
        (pl.lit(None, dtype=pl.Float64), lambda col: col.is_finite()),
        (pl.lit(None, dtype=pl.Float64), lambda col: col.is_infinite()),
    ],
)
def test_join_rewrite_null_preserving_exprs(
    expr_first_input: Any, expr_func: Callable[[pl.Expr], pl.Expr]
) -> None:
    lhs = pl.LazyFrame({"a": 1})
    rhs = pl.select(a=1, x=expr_first_input).lazy()

    assert (
        pl.select(expr_first_input)
        .select(expr_func(pl.first()))
        .select(pl.first().is_null() | ~pl.first())
        .to_series()
        .item()
    )

    q = lhs.join(rhs, on="a", how="left", maintain_order="left_right").filter(
        expr_func(pl.col("x"))
    )

    plan = q.explain()
    assert plan.startswith("INNER JOIN")

    out = q.collect()

    assert out.height == 0
    assert_frame_equal(out, q.collect(optimizations=pl.QueryOptFlags.none()))


@pytest.mark.parametrize(
    ("expr_first_input", "expr_func"),
    [
        (
            pl.lit(None, dtype=pl.Int64),
            lambda x: ~(x.is_in([1, None], nulls_equal=True)),
        ),
        (
            pl.lit(None, dtype=pl.Int64),
            lambda x: x.is_in([1, None], nulls_equal=True) > True,
        ),
        (
            pl.lit(None, dtype=pl.Int64),
            lambda x: x.is_in([1], nulls_equal=True),
        ),
    ],
)
def test_join_rewrite_forbid_exprs(
    expr_first_input: Any, expr_func: Callable[[pl.Expr], pl.Expr]
) -> None:
    lhs = pl.LazyFrame({"a": 1})
    rhs = pl.select(a=1, x=expr_first_input).lazy()

    q = lhs.join(rhs, on="a", how="left", maintain_order="left_right").filter(
        expr_func(pl.col("x"))
    )

    plan = q.explain()
    assert plan.startswith("FILTER")

    assert_frame_equal(q.collect(), q.collect(optimizations=pl.QueryOptFlags.none()))


def test_join_filter_pushdown_iejoin_cse_23469() -> None:
    lf_x = pl.LazyFrame({"x": [1, 2, 3]})
    lf_y = pl.LazyFrame({"y": [1, 2, 3]})

    lf_xy = lf_x.join(lf_y, how="cross").filter(pl.col("x") > pl.col("y"))

    q = pl.concat([lf_xy, lf_xy])

    assert_frame_equal(
        q.collect().sort(pl.all()),
        pl.DataFrame(
            {
                "x": [2, 2, 3, 3, 3, 3],
                "y": [1, 1, 1, 1, 2, 2],
            },
        ),
    )

    q = pl.concat([lf_xy, lf_xy]).filter(pl.col("x") > pl.col("y"))

    assert_frame_equal(
        q.collect().sort(pl.all()),
        pl.DataFrame(
            {
                "x": [2, 2, 3, 3, 3, 3],
                "y": [1, 1, 1, 1, 2, 2],
            },
        ),
    )

    q = (
        lf_x.join_where(lf_y, pl.col("x") == pl.col("y"))
        .cache()
        .filter(pl.col("x") >= 0)
    )

    assert_frame_equal(
        q.collect().sort(pl.all()), pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    )


def test_join_cast_type_coercion_23236() -> None:
    lhs = pl.LazyFrame([{"name": "a"}]).rename({"name": "newname"})
    rhs = pl.LazyFrame([{"name": "a"}])

    q = lhs.join(rhs, left_on=pl.col("newname").cast(pl.String), right_on="name")

    assert_frame_equal(q.collect(), pl.DataFrame({"newname": "a", "name": "a"}))


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        (
            "inner",
            pl.DataFrame(schema={"a": pl.Int128, "a_right": pl.Int128}),
        ),
        (
            "left",
            pl.DataFrame(
                {"a": [1, 1, 2], "a_right": None},
                schema={"a": pl.Int128, "a_right": pl.Int128},
            ),
        ),
        (
            "right",
            pl.DataFrame(
                {
                    "a": None,
                    "a_right": [
                        -9223372036854775808,
                        -9223372036854775807,
                        -9223372036854775806,
                    ],
                },
                schema={"a": pl.Int128, "a_right": pl.Int128},
            ),
        ),
        (
            "full",
            pl.DataFrame(
                [
                    pl.Series("a", [None, None, None, 1, 1, 2], dtype=pl.Int128),
                    pl.Series(
                        "a_right",
                        [
                            -9223372036854775808,
                            -9223372036854775807,
                            -9223372036854775806,
                            None,
                            None,
                            None,
                        ],
                        dtype=pl.Int128,
                    ),
                ]
            ),
        ),
        (
            "semi",
            pl.DataFrame([pl.Series("a", [], dtype=pl.Int128)]),
        ),
        (
            "anti",
            pl.DataFrame([pl.Series("a", [1, 1, 2], dtype=pl.Int128)]),
        ),
    ],
)
@pytest.mark.parametrize(
    ("sort_left", "sort_right"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_join_i128_23688(
    how: str, expected: pl.DataFrame, sort_left: bool, sort_right: bool
) -> None:
    lhs = pl.LazyFrame({"a": pl.Series([1, 1, 2], dtype=pl.Int128)})

    rhs = pl.LazyFrame(
        {
            "a": pl.Series(
                [
                    -9223372036854775808,
                    -9223372036854775807,
                    -9223372036854775806,
                ],
                dtype=pl.Int128,
            )
        }
    )

    lhs = lhs.collect().sort("a").lazy() if sort_left else lhs
    rhs = rhs.collect().sort("a").lazy() if sort_right else rhs

    q = lhs.join(rhs, on="a", how=how, coalesce=False)  # type: ignore[arg-type]

    assert_frame_equal(
        q.collect().sort(pl.all()),
        expected,
    )

    q = (
        lhs.with_columns(b=pl.col("a"))
        .join(
            rhs.with_columns(b=pl.col("a")),
            on=["a", "b"],
            how=how,  # type: ignore[arg-type]
            coalesce=False,
        )
        .select(expected.columns)
    )

    assert_frame_equal(
        q.collect().sort(pl.all()),
        expected,
    )


def test_join_asof_by_i128() -> None:
    lhs = pl.LazyFrame({"a": pl.Series([1, 1, 2], dtype=pl.Int128), "i": 1})

    rhs = pl.LazyFrame(
        {
            "a": pl.Series(
                [
                    -9223372036854775808,
                    -9223372036854775807,
                    -9223372036854775806,
                ],
                dtype=pl.Int128,
            ),
            "i": 1,
        }
    ).with_columns(b=pl.col("a"))

    q = lhs.join_asof(rhs, on="i", by="a")

    assert_frame_equal(
        q.collect().sort(pl.all()),
        pl.DataFrame(
            {"a": [1, 1, 2], "i": 1, "b": None},
            schema={"a": pl.Int128, "i": pl.Int32, "b": pl.Int128},
        ),
    )
