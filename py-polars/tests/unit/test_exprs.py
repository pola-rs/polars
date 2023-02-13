from __future__ import annotations

import random
import typing
from typing import cast

import numpy as np
import pytest

import polars as pl
from polars.datatypes import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    TEMPORAL_DTYPES,
)
from polars.testing import assert_frame_equal, assert_series_equal


def test_horizontal_agg(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(pl.max([pl.col("A"), pl.col("B")]))
    assert out[:, 0].to_list() == [5, 4, 3, 4, 5]

    out = df.select(pl.min([pl.col("A"), pl.col("B")]))
    assert out[:, 0].to_list() == [1, 2, 3, 2, 1]


def test_suffix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().suffix("_reverse")])
    assert out.columns == ["A_reverse", "fruits_reverse", "B_reverse", "cars_reverse"]


def test_prefix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().prefix("reverse_")])
    assert out.columns == ["reverse_A", "reverse_fruits", "reverse_B", "reverse_cars"]


def test_cumcount() -> None:
    df = pl.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], schema=["A"])

    out = df.groupby("A", maintain_order=True).agg(
        [pl.col("A").cumcount(reverse=False).alias("foo")]
    )

    assert out["foo"][0].to_list() == [0, 1, 2, 3]
    assert out["foo"][1].to_list() == [0, 1]


def test_filter_where() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 1, 2, 3], "b": [4, 5, 6, 7, 8, 9]})
    result_where = df.groupby("a", maintain_order=True).agg(
        pl.col("b").where(pl.col("b") > 4).alias("c")
    )
    result_filter = df.groupby("a", maintain_order=True).agg(
        pl.col("b").filter(pl.col("b") > 4).alias("c")
    )
    expected = pl.DataFrame({"a": [1, 2, 3], "c": [[7], [5, 8], [6, 9]]})
    assert_frame_equal(result_where, expected)
    assert_frame_equal(result_filter, expected)


def test_min_nulls_consistency() -> None:
    df = pl.DataFrame({"a": [None, 2, 3], "b": [4, None, 6], "c": [7, 5, 0]})
    out = df.select([pl.min(["a", "b", "c"])]).to_series()
    expected = pl.Series("min", [4, 2, 0])
    assert_series_equal(out, expected)

    out = df.select([pl.max(["a", "b", "c"])]).to_series()
    expected = pl.Series("max", [7, 5, 6])
    assert_series_equal(out, expected)


def test_list_join_strings() -> None:
    s = pl.Series("a", [["ab", "c", "d"], ["e", "f"], ["g"], []])
    expected = pl.Series("a", ["ab-c-d", "e-f", "g", ""])
    assert_series_equal(s.arr.join("-"), expected)


def test_count_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 3, 3], "b": ["a", "a", "b", "a", "a"]})

    out = df.select(pl.count())
    assert out.shape == (1, 1)
    assert cast(int, out.item()) == 5

    out = df.groupby("b", maintain_order=True).agg(pl.count())
    assert out["b"].to_list() == ["a", "b"]
    assert out["count"].to_list() == [4, 1]


def test_shuffle() -> None:
    # Setting random.seed should lead to reproducible results
    s = pl.Series("a", range(20))
    random.seed(1)
    result1 = pl.select(pl.lit(s).shuffle()).to_series()
    random.seed(1)
    result2 = pl.select(pl.lit(s).shuffle()).to_series()
    assert_series_equal(result1, result2)


def test_sample() -> None:
    a = pl.Series("a", range(0, 20))
    out = pl.select(
        pl.lit(a).sample(frac=0.5, with_replacement=False, seed=1)
    ).to_series()

    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)
    assert set(out).issubset(set(a))

    out = pl.select(pl.lit(a).sample(n=10, with_replacement=False, seed=1)).to_series()
    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)

    # Setting random.seed should lead to reproducible results
    random.seed(1)
    result1 = pl.select(pl.lit(a).sample(n=10)).to_series()
    random.seed(1)
    result2 = pl.select(pl.lit(a).sample(n=10)).to_series()
    assert_series_equal(result1, result2)


def test_map_alias() -> None:
    out = pl.DataFrame({"foo": [1, 2, 3]}).select(
        (pl.col("foo") * 2).map_alias(lambda name: f"{name}{name}")
    )
    expected = pl.DataFrame({"foofoo": [2, 4, 6]})
    assert_frame_equal(out, expected)


def test_unique_stable() -> None:
    s = pl.Series("a", [1, 1, 1, 1, 2, 2, 2, 3, 3])
    expected = pl.Series("a", [1, 2, 3])
    assert_series_equal(s.unique(maintain_order=True), expected)


def test_unique_and_drop_stability() -> None:
    # see: 2898
    # the original cause was that we wrote:
    # expr_a = a.unique()
    # expr_a.filter(a.unique().is_not_null())
    # meaning that the a.unique was executed twice, which is an unstable algorithm
    df = pl.DataFrame({"a": [1, None, 1, None]})
    assert df.select(pl.col("a").unique().drop_nulls()).to_series()[0] == 1


def test_unique_counts() -> None:
    s = pl.Series("id", ["a", "b", "b", "c", "c", "c"])
    expected = pl.Series("id", [1, 2, 3], dtype=pl.UInt32)
    assert_series_equal(s.unique_counts(), expected)


def test_entropy() -> None:
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "B"],
            "id": [1, 2, 1, 4, 5, 4, 6],
        }
    )
    result = df.groupby("group", maintain_order=True).agg(
        pl.col("id").entropy(normalize=True)
    )
    expected = pl.DataFrame(
        {"group": ["A", "B"], "id": [1.0397207708399179, 1.371381017771811]}
    )
    assert_frame_equal(result, expected)


def test_dot_in_groupby() -> None:
    df = pl.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "x": [1, 1, 1, 1, 1, 1],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    result = df.groupby("group", maintain_order=True).agg(
        pl.col("x").dot("y").alias("dot")
    )
    expected = pl.DataFrame({"group": ["a", "b"], "dot": [6, 15]})
    assert_frame_equal(result, expected)


def test_dtype_col_selection() -> None:
    df = pl.DataFrame(
        data=[],
        schema={
            "a1": pl.Datetime,
            "a2": pl.Datetime("ms"),
            "a3": pl.Datetime("ms"),
            "a4": pl.Datetime("ns"),
            "b": pl.Date,
            "c": pl.Time,
            "d1": pl.Duration,
            "d2": pl.Duration("ms"),
            "d3": pl.Duration("us"),
            "d4": pl.Duration("ns"),
            "e": pl.Int8,
            "f": pl.Int16,
            "g": pl.Int32,
            "h": pl.Int64,
            "i": pl.Float32,
            "j": pl.Float64,
            "k": pl.UInt8,
            "l": pl.UInt16,
            "m": pl.UInt32,
            "n": pl.UInt64,
        },
    )
    assert df.select(pl.col(INTEGER_DTYPES)).columns == [
        "e",
        "f",
        "g",
        "h",
        "k",
        "l",
        "m",
        "n",
    ]
    assert df.select(pl.col(FLOAT_DTYPES)).columns == ["i", "j"]
    assert df.select(pl.col(NUMERIC_DTYPES)).columns == [
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
    ]
    assert df.select(pl.col(TEMPORAL_DTYPES)).columns == [
        "a1",
        "a2",
        "a3",
        "a4",
        "b",
        "c",
        "d1",
        "d2",
        "d3",
        "d4",
    ]
    assert df.select(pl.col(DATETIME_DTYPES)).columns == [
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    assert df.select(pl.col(DURATION_DTYPES)).columns == [
        "d1",
        "d2",
        "d3",
        "d4",
    ]


def test_list_eval_expression() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})

    for parallel in [True, False]:
        assert df.with_columns(
            pl.concat_list(["a", "b"])
            .arr.eval(pl.first().rank(), parallel=parallel)
            .alias("rank")
        ).to_dict(False) == {
            "a": [1, 8, 3],
            "b": [4, 5, 2],
            "rank": [[1.0, 2.0], [2.0, 1.0], [2.0, 1.0]],
        }

        assert df["a"].reshape((1, -1)).arr.eval(
            pl.first(), parallel=parallel
        ).to_list() == [[1, 8, 3]]


def test_null_count_expr() -> None:
    df = pl.DataFrame({"key": ["a", "b", "b", "a"], "val": [1, 2, None, 1]})

    assert df.select([pl.all().null_count()]).to_dict(False) == {"key": [0], "val": [1]}


def test_power_by_expression() -> None:
    out = pl.DataFrame(
        {"a": [1, None, None, 4, 5, 6], "b": [1, 2, None, 4, None, 6]}
    ).select(
        [
            (pl.col("a") ** pl.col("b")).alias("pow"),
            (2 ** pl.col("b")).alias("pow_left"),
        ]
    )

    assert out["pow"].to_list() == [
        1.0,
        None,
        None,
        256.0,
        None,
        46656.0,
    ]
    assert out["pow_left"].to_list() == [
        2.0,
        4.0,
        None,
        16.0,
        None,
        64.0,
    ]


def test_expression_appends() -> None:
    df = pl.DataFrame({"a": [1, 1, 2]})

    assert df.select(pl.repeat(None, 3).append(pl.col("a"))).n_chunks() == 2

    assert df.select(pl.repeat(None, 3).append(pl.col("a")).rechunk()).n_chunks() == 1

    out = df.select(pl.concat([pl.repeat(None, 3), pl.col("a")]))

    assert out.n_chunks() == 1
    assert out.to_series().to_list() == [None, None, None, 1, 1, 2]


def test_regex_in_filter() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, None, 5],
            "names": ["foo", "ham", "spam", "egg", None],
            "flt": [1.0, None, 3.0, 1.0, None],
        }
    )

    res = df.filter(
        pl.fold(acc=False, f=lambda acc, s: acc | s, exprs=(pl.col("^nrs|flt*$") < 3))
    ).row(0)
    expected = (1, "foo", 1.0)
    assert res == expected


def test_arr_contains() -> None:
    df_groups = pl.DataFrame(
        {
            "str_list": [
                ["cat", "mouse", "dog"],
                ["dog", "mouse", "cat"],
                ["dog", "mouse", "aardvark"],
            ],
        }
    )
    assert df_groups.lazy().filter(
        pl.col("str_list").arr.contains("cat")
    ).collect().to_dict(False) == {
        "str_list": [["cat", "mouse", "dog"], ["dog", "mouse", "cat"]]
    }


def test_rank_so_4109() -> None:
    # also tests ranks null behavior
    df = pl.from_dict(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "rank": [None, 3, 2, 4, 1, 4, 3, 2, 1, None, 3, 4, 4, 1, None, 3],
        }
    ).sort(by=["id", "rank"])

    assert df.groupby("id").agg(
        [
            pl.col("rank").alias("original"),
            pl.col("rank").rank(method="dense").alias("dense"),
            pl.col("rank").rank(method="average").alias("average"),
        ]
    ).to_dict(False) == {
        "id": [1, 2, 3, 4],
        "original": [[None, 2, 3, 4], [1, 2, 3, 4], [None, 1, 3, 4], [None, 1, 3, 4]],
        "dense": [[None, 1, 2, 3], [1, 2, 3, 4], [None, 1, 2, 3], [None, 1, 2, 3]],
        "average": [
            [None, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [None, 1.0, 2.0, 3.0],
            [None, 1.0, 2.0, 3.0],
        ],
    }


def test_unique_empty() -> None:
    for dt in [pl.Utf8, pl.Boolean, pl.Int32, pl.UInt32]:
        s = pl.Series([], dtype=dt)
        assert_series_equal(s.unique(), s)


@typing.no_type_check
def test_search_sorted() -> None:
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        a = np.sort(np.random.randn(10) * 100)
        s = pl.Series(a)

        for v in range(int(np.min(a)), int(np.max(a)), 20):
            assert np.searchsorted(a, v) == s.search_sorted(v)

    a = pl.Series([1, 2, 3])
    b = pl.Series([1, 2, 2, -1])
    assert a.search_sorted(b).to_list() == [0, 1, 1, 0]
    b = pl.Series([1, 2, 2, None, 3])
    assert a.search_sorted(b).to_list() == [0, 1, 1, 0, 2]

    a = pl.Series(["b", "b", "d", "d"])
    b = pl.Series(["a", "b", "c", "d", "e"])
    assert a.search_sorted(b, side="left").to_list() == [0, 0, 2, 2, 4]
    assert a.search_sorted(b, side="right").to_list() == [0, 2, 2, 4, 4]

    a = pl.Series([1, 1, 4, 4])
    b = pl.Series([0, 1, 2, 4, 5])
    assert a.search_sorted(b, side="left").to_list() == [0, 0, 2, 2, 4]
    assert a.search_sorted(b, side="right").to_list() == [0, 2, 2, 4, 4]


def test_abs_expr() -> None:
    df = pl.DataFrame({"x": [-1, 0, 1]})
    out = df.select(abs(pl.col("x")))

    assert out["x"].to_list() == [1, 0, 1]


def test_logical_boolean() -> None:
    # note, cannot use expressions in logical
    # boolean context (eg: and/or/not operators)
    with pytest.raises(ValueError, match="ambiguous"):
        pl.col("colx") and pl.col("coly")

    with pytest.raises(ValueError, match="ambiguous"):
        pl.col("colx") or pl.col("coly")


# https://github.com/pola-rs/polars/issues/4951
def test_ewm_with_multiple_chunks() -> None:
    df0 = pl.DataFrame(
        data=[
            ("w", 6.0, 1.0),
            ("x", 5.0, 2.0),
            ("y", 4.0, 3.0),
            ("z", 3.0, 4.0),
        ],
        schema=["a", "b", "c"],
    ).with_columns(
        [
            pl.col(pl.Float64).log().diff().prefix("ld_"),
        ]
    )
    assert df0.n_chunks() == 1

    # NOTE: We aren't testing whether `select` creates two chunks;
    # we just need two chunks to properly test `ewm_mean`
    df1 = df0.select(["ld_b", "ld_c"])
    assert df1.n_chunks() == 2

    ewm_std = df1.with_columns(
        [
            pl.all().ewm_std(com=20).prefix("ewm_"),
        ]
    )
    assert ewm_std.null_count().sum(axis=1)[0] == 4


def test_map_dict() -> None:
    country_code_dict = {
        "CA": "Canada",
        "DE": "Germany",
        "FR": "France",
        None: "Not specified",
    }
    df = pl.DataFrame(
        {
            "country_code": ["FR", None, "ES", "DE"],
        }
    ).with_row_count()

    assert (
        df.with_columns(
            pl.col("country_code")
            .map_dict(country_code_dict, default=pl.col("country_code"))
            .alias("remapped")
        )
    ).to_dict(False) == {
        "row_nr": [0, 1, 2, 3],
        "country_code": ["FR", None, "ES", "DE"],
        "remapped": ["France", "Not specified", "ES", "Germany"],
    }

    assert (
        df.with_columns(
            pl.col("country_code").map_dict(country_code_dict).alias("remapped")
        )
    ).to_dict(False) == {
        "row_nr": [0, 1, 2, 3],
        "country_code": ["FR", None, "ES", "DE"],
        "remapped": ["France", "Not specified", None, "Germany"],
    }

    assert (
        df.with_columns(
            pl.struct(pl.col(["country_code", "row_nr"]))  # type: ignore[union-attr]
            .map_dict(
                country_code_dict,
                default=pl.col("row_nr").cast(pl.Utf8),
            )
            .alias("remapped")
        )
    ).to_dict(False) == {
        "row_nr": [0, 1, 2, 3],
        "country_code": ["FR", None, "ES", "DE"],
        "remapped": ["France", "Not specified", "2", "Germany"],
    }
