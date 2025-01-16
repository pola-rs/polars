from __future__ import annotations

import typing
import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from polars._typing import JoinStrategy


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

    np.testing.assert_equal(df.join(other=other, on="a").to_numpy(), result.to_numpy())
    np.testing.assert_equal(
        df.lazy().join(other=other.lazy(), on="a").collect().to_numpy(),
        result.to_numpy(),
    )


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
        df_b.lazy(), on=pl.col("a").cast(pl.Int64)
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
    assert (
        df1.join(df2, how="cross").join(
            df3,
            on=["index1", "index2", "index3"],
            how="left",
        )
    ).to_dict(as_series=False) == {
        "index1": [0, 0, 1, 1],
        "index2": [10, 10, 11, 11],
        "index3": [100, 101, 100, 101],
    }

    assert (
        df1.join(df2, how="cross").join(
            df3,
            on=["index3", "index1", "index2"],
            how="left",
        )
    ).to_dict(as_series=False) == {
        "index1": [0, 0, 1, 1],
        "index2": [10, 10, 11, 11],
        "index3": [100, 101, 100, 101],
    }


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

    with pytest.raises(TypeError, match="expected `other`.* LazyFrame"):
        _ = ldf.join(df, on="A")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected `other`.* DataFrame"):
        _ = df.join(ldf, on="A")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected `other`.* LazyFrame"):
        _ = ldf.join_asof(df, on="A")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected `other`.* DataFrame"):
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

    df1 = pl.DataFrame({"id": [1], "vals": [[]]}, schema=schema)  # type: ignore[arg-type]
    df2 = pl.DataFrame({"id": [2, 3], "vals": [[], [4]]}, schema=schema)  # type: ignore[arg-type]
    assert df1.join(df2, on="id", how="full").to_dict(as_series=False) == {
        "id": [None, None, 1],
        "vals": [None, None, []],
        "id_right": [2, 3, None],
        "vals_right": [[], [4.0], None],
    }


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
    assert df1.join(df2, on="id", how="full").to_dict(as_series=False) == {
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
    assert_frame_equal(out, expected)


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
    lhs = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [1, 2, 3],
            "c_right": [1, 2, 3],
        }
    )
    rhs = lhs.clone()
    with pytest.raises(DuplicateError, match="'c_right' already exists"):
        lhs.join(rhs, on=["a", "b"], how="left")


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

    assert a.join(a, on=["a", "b", "c", "d"], how="inner", join_nulls=True).shape == (
        644,
        4,
    )
    assert a.join(a, on=["a", "b", "c", "d"], how="inner", join_nulls=False).shape == (
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

    assert symbols.join(dates, left_on=pl.lit(1), right_on=pl.lit(1)).collect().to_dict(
        as_series=False
    ) == {"symbol": [4, 4, 4, 5, 5, 5, 6, 6, 6], "date": [1, 2, 3, 1, 2, 3, 1, 2, 3]}


def test_join_empty_literal_17027() -> None:
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame(schema={"a": pl.Int64})

    assert df1.join(df2, on=pl.lit(0), how="left").height == 1
    assert df1.join(df2, on=pl.lit(0), how="inner").height == 0
    assert (
        df1.lazy()
        .join(df2.lazy(), on=pl.lit(0), how="inner")
        .collect(streaming=True)
        .height
        == 0
    )
    assert (
        df1.lazy()
        .join(df2.lazy(), on=pl.lit(0), how="left")
        .collect(streaming=True)
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
    a = pl.LazyFrame(
        {"a": [1], "c": [None]}, schema={"a": pl.Int64, "c": pl.Categorical}
    )
    b = pl.LazyFrame({"a": [1, 3, 4]})
    assert a.join(b, on="a", how="full", coalesce=True).collect().to_dict(
        as_series=False
    ) == {"a": [1, 3, 4], "c": [None, None, None]}


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
def test_join_numeric_type_upcast_15338(
    dtypes: tuple[str, str, str], swap: bool
) -> None:
    supertype, ltype, rtype = (getattr(pl, x) for x in dtypes)
    ltype, rtype = (rtype, ltype) if swap else (ltype, rtype)

    left = pl.select(pl.Series("a", [1, 1, 3]).cast(ltype)).lazy()
    right = pl.select(pl.Series("a", [1]).cast(rtype), b=pl.lit("A")).lazy()

    assert_frame_equal(
        left.join(right, on="a", how="left").collect(),
        pl.select(a=pl.Series([1, 1, 3]).cast(ltype), b=pl.Series(["A", "A", None])),
    )

    assert_frame_equal(
        left.join(right, on="a", how="left", coalesce=False).drop("a_right").collect(),
        pl.select(a=pl.Series([1, 1, 3]).cast(ltype), b=pl.Series(["A", "A", None])),
    )

    assert_frame_equal(
        left.join(right, on="a", how="full").collect(),
        pl.select(
            a=pl.Series([1, 1, 3]).cast(ltype),
            a_right=pl.Series([1, 1, None]).cast(rtype),
            b=pl.Series(["A", "A", None]),
        ),
    )

    assert_frame_equal(
        left.join(right, on="a", how="full", coalesce=True).collect(),
        pl.select(
            a=pl.Series([1, 1, 3]).cast(supertype),
            b=pl.Series(["A", "A", None]),
        ),
    )

    assert_frame_equal(
        left.join(right, on="a", how="semi").collect(),
        pl.select(a=pl.Series([1, 1]).cast(ltype)),
    )


def test_join_numeric_type_upcast_forbid_float_int() -> None:
    ltype = pl.Float64
    rtype = pl.Int32

    left = pl.LazyFrame(schema={"a": ltype})
    right = pl.LazyFrame(schema={"a": rtype})

    with pytest.raises(SchemaError, match="datatypes of join keys don't match"):
        left.join(right, on="a", how="left").collect()


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
