from __future__ import annotations

import typing
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_series_equal
from polars.testing._private import verify_series_and_expr_api


def test_list_arr_get() -> None:
    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.arr.get(0)
    expected = pl.Series("a", [1, 4, 6])
    assert_series_equal(out, expected)
    out = a.arr[0]
    expected = pl.Series("a", [1, 4, 6])
    assert_series_equal(out, expected)
    out = a.arr.first()
    assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).arr.first()).to_series()
    assert_series_equal(out, expected)

    out = a.arr.get(-1)
    expected = pl.Series("a", [3, 5, 9])
    assert_series_equal(out, expected)
    out = a.arr.last()
    assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).arr.last()).to_series()
    assert_series_equal(out, expected)

    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.arr.get(-3)
    expected = pl.Series("a", [1, None, 7])
    assert_series_equal(out, expected)

    assert pl.DataFrame(
        {"a": [[1], [2], [3], [4, 5, 6], [7, 8, 9], [None, 11]]}
    ).with_columns(
        [pl.col("a").arr.get(i).alias(f"get_{i}") for i in range(4)]
    ).to_dict(
        False
    ) == {
        "a": [[1], [2], [3], [4, 5, 6], [7, 8, 9], [None, 11]],
        "get_0": [1, 2, 3, 4, 7, None],
        "get_1": [None, None, None, 5, 8, 11],
        "get_2": [None, None, None, 6, 9, None],
        "get_3": [None, None, None, None, None, None],
    }

    # get by indexes where some are out of bounds
    df = pl.DataFrame({"cars": [[1, 2, 3], [2, 3], [4], []], "indexes": [-2, 1, -3, 0]})

    assert df.select([pl.col("cars").arr.get("indexes")]).to_dict(False) == {
        "cars": [2, 3, None, None]
    }


def test_contains() -> None:
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    out = a.arr.contains(2)
    expected = pl.Series("a", [True, True, False])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).arr.contains(2)).to_series()
    assert_series_equal(out, expected)


def test_dtype() -> None:
    # inferred
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    assert a.dtype == pl.List
    assert a.inner_dtype == pl.Int64
    assert a.dtype.inner == pl.Int64  # type: ignore[attr-defined]

    # explicit
    df = pl.DataFrame(
        data={
            "i": [[1, 2, 3]],
            "tm": [[time(10, 30, 45)]],
            "dt": [[date(2022, 12, 31)]],
            "dtm": [[datetime(2022, 12, 31, 1, 2, 3)]],
        },
        columns=[
            ("i", pl.List(pl.Int8)),
            ("tm", pl.List(pl.Time)),
            ("dt", pl.List(pl.Date)),
            ("dtm", pl.List(pl.Datetime)),
        ],
    )
    assert df.schema == {
        "i": pl.List(pl.Int8),
        "tm": pl.List(pl.Time),
        "dt": pl.List(pl.Date),
        "dtm": pl.List(pl.Datetime),
    }
    assert df.schema["i"].inner == pl.Int8  # type: ignore[union-attr]
    assert df.rows() == [
        (
            [1, 2, 3],
            [time(10, 30, 45)],
            [date(2022, 12, 31)],
            [datetime(2022, 12, 31, 1, 2, 3)],
        )
    ]


def test_categorical() -> None:
    # https://github.com/pola-rs/polars/issues/2038
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 1, 1, 1, 1, 1, 1]),
            pl.Series("b", [8, 2, 3, 6, 3, 6, 2, 2]),
            pl.Series("c", ["a", "b", "c", "a", "b", "c", "a", "b"]).cast(
                pl.Categorical
            ),
        ]
    )
    out = (
        df.groupby(["a", "b"])
        .agg(
            [
                pl.col("c").count().alias("num_different_c"),
                pl.col("c").alias("c_values"),
            ]
        )
        .filter(pl.col("num_different_c") >= 2)
        .to_series(3)
    )

    assert out.inner_dtype == pl.Categorical


def test_list_concat_rolling_window() -> None:
    # inspired by:
    # https://stackoverflow.com/questions/70377100/use-the-rolling-function-of-polars-to-get-a-list-of-all-values-in-the-rolling-wi
    # this tests if it works without specifically creating list dtype upfront. note that
    # the given answer is preferred over this snippet as that reuses the list array when
    # shifting
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        }
    )
    out = df.with_columns(
        [pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)]
    ).select(
        [pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")]
    )
    assert out.shape == (5, 1)

    s = out.to_series()
    assert s.dtype == pl.List
    assert s.to_list() == [
        [None, None, 1.0],
        [None, 1.0, 2.0],
        [1.0, 2.0, 9.0],
        [2.0, 9.0, 2.0],
        [9.0, 2.0, 13.0],
    ]

    # this test proper null behavior of concat list
    out = (
        df.with_column(pl.col("A").reshape((-1, 1)))  # first turn into a list
        .with_columns(
            [
                pl.col("A").shift(i).alias(f"A_lag_{i}")
                for i in range(3)  # slice the lists to a lag
            ]
        )
        .select(
            [
                pl.all(),
                pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias(
                    "A_rolling"
                ),
            ]
        )
    )
    assert out.shape == (5, 5)

    l64 = pl.List(pl.Float64)
    assert out.schema == {
        "A": l64,
        "A_lag_0": l64,
        "A_lag_1": l64,
        "A_lag_2": l64,
        "A_rolling": l64,
    }


def test_list_append() -> None:
    df = pl.DataFrame({"a": [[1, 2], [1], [1, 2, 3]]})

    out = df.select([pl.col("a").arr.concat(pl.Series([[1, 2]]))])
    assert out["a"][0].to_list() == [1, 2, 1, 2]

    out = df.select([pl.col("a").arr.concat([1, 4])])
    assert out["a"][0].to_list() == [1, 2, 1, 4]

    out_s = df["a"].arr.concat([4, 1])
    assert out_s[0].to_list() == [1, 2, 4, 1]


def test_list_arr_empty() -> None:
    df = pl.DataFrame({"cars": [[1, 2, 3], [2, 3], [4], []]})

    out = df.select(
        [
            pl.col("cars").arr.first().alias("cars_first"),
            pl.when(pl.col("cars").arr.first() == 2)
            .then(1)
            .when(pl.col("cars").arr.contains(2))
            .then(2)
            .otherwise(3)
            .alias("cars_literal"),
        ]
    )
    expected = pl.DataFrame(
        {"cars_first": [1, 2, 4, None], "cars_literal": [2, 1, 3, 3]}
    )
    assert out.frame_equal(expected)


def test_list_argminmax() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [0, 2], dtype=pl.UInt32)
    verify_series_and_expr_api(s, expected, "arr.arg_min")
    expected = pl.Series("a", [1, 0], dtype=pl.UInt32)
    verify_series_and_expr_api(s, expected, "arr.arg_max")


def test_list_shift() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [[None, 1], [None, 3, 2]])
    assert s.arr.shift().to_list() == expected.to_list()


def test_list_diff() -> None:
    s = pl.Series("a", [[1, 2], [10, 2, 1]])
    expected = pl.Series("a", [[None, 1], [None, -8, -1]])
    assert s.arr.diff().to_list() == expected.to_list()


def test_slice() -> None:
    vals = [[1, 2, 3, 4], [10, 2, 1]]
    s = pl.Series("a", vals)
    assert s.arr.head(2).to_list() == [[1, 2], [10, 2]]
    assert s.arr.tail(2).to_list() == [[3, 4], [2, 1]]
    assert s.arr.tail(200).to_list() == vals
    assert s.arr.head(200).to_list() == vals
    assert s.arr.slice(1, 2).to_list() == [[2, 3], [2, 1]]


def test_cast_inner() -> None:
    a = pl.Series([[1, 2]])
    for t in [bool, pl.Boolean]:
        b = a.cast(pl.List(t))
        assert b.dtype == pl.List(pl.Boolean)
        assert b.to_list() == [[True, True]]

    # this creates an inner null type
    df = pl.from_pandas(pd.DataFrame(data=[[[]], [[]]], columns=["A"]))
    assert (
        df["A"].cast(pl.List(int)).dtype.inner == pl.Int64  # type: ignore[attr-defined]
    )


def test_list_eval_dtype_inference() -> None:
    grades = pl.DataFrame(
        {
            "student": ["bas", "laura", "tim", "jenny"],
            "arithmetic": [10, 5, 6, 8],
            "biology": [4, 6, 2, 7],
            "geography": [8, 4, 9, 7],
        }
    )

    rank_pct = pl.col("").rank(reverse=True) / pl.col("").count().cast(pl.UInt16)

    # the .arr.first() would fail if .arr.eval did not correctly infer the output type
    assert grades.with_column(
        pl.concat_list(pl.all().exclude("student")).alias("all_grades")
    ).select(
        [
            pl.col("all_grades")
            .arr.eval(rank_pct, parallel=True)
            .alias("grades_rank")
            .arr.first()
        ]
    ).to_series().to_list() == [
        0.3333333432674408,
        0.6666666865348816,
        0.6666666865348816,
        0.3333333432674408,
    ]


def test_list_empty_groupby_result_3521() -> None:
    # Create a left relation where the join column contains a null value
    left = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("groupby_column"),
            pl.lit(None).cast(pl.Int32).alias("join_column"),
        ]
    )

    # Create a right relation where there is a column to count distinct on
    right = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("join_column"),
            pl.lit(1).alias("n_unique_column"),
        ]
    )

    # Calculate n_unique after dropping nulls
    # This will panic on polars version 0.13.38 and 0.13.39
    assert (
        left.join(right, on="join_column", how="left")
        .groupby("groupby_column")
        .agg(pl.col("n_unique_column").drop_nulls())
    ).to_dict(False) == {"groupby_column": [1], "n_unique_column": [[]]}


def test_list_fill_null() -> None:
    df = pl.DataFrame({"C": [["a", "b", "c"], [], [], ["d", "e"]]})
    assert df.with_columns(
        [
            pl.when(pl.col("C").arr.lengths() == 0)
            .then(None)
            .otherwise(pl.col("C"))
            .alias("C")
        ]
    ).to_series().to_list() == [["a", "b", "c"], None, None, ["d", "e"]]


def test_list_fill_list() -> None:
    assert pl.DataFrame({"a": [[1, 2, 3], []]}).select(
        [
            pl.when(pl.col("a").arr.lengths() == 0)
            .then([5])
            .otherwise(pl.col("a"))
            .alias("filled")
        ]
    ).to_dict(False) == {"filled": [[1, 2, 3], [5]]}


def test_empty_list_construction() -> None:
    assert pl.Series([[]]).to_list() == [[]]
    assert pl.DataFrame([{"array": [], "not_array": 1234}], orient="row").to_dict(
        False
    ) == {"array": [[]], "not_array": [1234]}

    df = pl.DataFrame(columns=[("col", pl.List)])
    assert df.schema == {"col": pl.List}
    assert df.rows() == []


def test_list_ternary_concat() -> None:
    df = pl.DataFrame(
        {
            "list1": [["123", "456"], None],
            "list2": [["789"], ["zzz"]],
        }
    )

    assert df.with_column(
        pl.when(pl.col("list1").is_null())
        .then(pl.col("list1").arr.concat(pl.col("list2")))
        .otherwise(pl.col("list2"))
        .alias("result")
    ).to_dict(False) == {
        "list1": [["123", "456"], None],
        "list2": [["789"], ["zzz"]],
        "result": [["789"], None],
    }

    assert df.with_column(
        pl.when(pl.col("list1").is_null())
        .then(pl.col("list2"))
        .otherwise(pl.col("list1").arr.concat(pl.col("list2")))
        .alias("result")
    ).to_dict(False) == {
        "list1": [["123", "456"], None],
        "list2": [["789"], ["zzz"]],
        "result": [["123", "456", "789"], ["zzz"]],
    }


def test_list_concat_nulls() -> None:
    assert pl.DataFrame(
        {
            "a": [["a", "b"], None, ["c", "d", "e"], None],
            "t": [["x"], ["y"], None, None],
        }
    ).with_column(pl.concat_list(["a", "t"]).alias("concat"))["concat"].to_list() == [
        ["a", "b", "x"],
        None,
        None,
        None,
    ]


def test_list_concat_supertype() -> None:
    df = pl.DataFrame(
        [pl.Series("a", [1, 2], pl.UInt8), pl.Series("b", [10000, 20000], pl.UInt16)]
    )
    assert df.with_column(pl.concat_list(pl.col(["a", "b"])).alias("concat_list"))[
        "concat_list"
    ].to_list() == [[1, 10000], [2, 20000]]


def test_list_hash() -> None:
    out = pl.DataFrame({"a": [[1, 2, 3], [3, 4], [1, 2, 3]]}).with_column(
        pl.col("a").hash().alias("b")
    )
    assert out.dtypes == [pl.List(pl.Int64), pl.UInt64]
    assert out[0, "b"] == out[2, "b"]


def test_arr_contains_categorical() -> None:
    df = pl.DataFrame(
        {"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]}
    ).lazy()
    df = df.with_column(pl.col("str").cast(pl.Categorical))
    df_groups = df.groupby("group").agg([pl.col("str").list().alias("str_list")])
    assert df_groups.filter(pl.col("str_list").arr.contains("C")).collect().to_dict(
        False
    ) == {"group": [2], "str_list": [["A", "C"]]}


def test_list_diagonal_concat() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})

    df2 = pl.DataFrame({"b": [[1]]})

    assert pl.concat([df1, df2], how="diagonal").to_dict(False) == {
        "a": [1, 2, None],
        "b": [None, None, [1]],
    }


def test_list_eval_type_coercion() -> None:
    last_non_null_value = pl.element().fill_null(3).last()
    df = pl.DataFrame(
        {
            "array_cols": [[1, None]],
        }
    )

    assert df.select(
        [
            pl.col("array_cols")
            .arr.eval(last_non_null_value, parallel=False)
            .alias("col_last")
        ]
    ).to_dict(False) == {"col_last": [[3]]}


def test_is_in_empty_list_4559() -> None:
    assert pl.Series(["a"]).is_in([]).to_list() == [False]


def test_is_in_empty_list_4639() -> None:
    df = pl.DataFrame({"a": [1, None]})
    empty_list: list[int] = []

    assert df.with_columns([pl.col("a").is_in(empty_list).alias("a_in_list")]).to_dict(
        False
    ) == {"a": [1, None], "a_in_list": [False, False]}
    df = pl.DataFrame()
    assert df.with_columns(
        [pl.lit(None).cast(pl.Int64).is_in(empty_list).alias("in_empty_list")]
    ).to_dict(False) == {"in_empty_list": [False]}


def test_inner_type_categorical_on_rechunk() -> None:
    df = pl.DataFrame({"cats": ["foo", "bar"]}).select(
        pl.col(pl.Utf8).cast(pl.Categorical).list()
    )

    assert pl.concat([df, df], rechunk=True).dtypes == [pl.List(pl.Categorical)]


def test_groupby_list_column() -> None:
    df = (
        pl.DataFrame({"a": ["a", "b", "a"]})
        .with_column(pl.col("a").cast(pl.Categorical))
        .groupby("a", maintain_order=True)
        .agg(pl.col("a").list().alias("a_list"))
    )

    assert df.groupby("a_list", maintain_order=True).first().to_dict(False) == {
        "a_list": [["a", "a"], ["b"]],
        "a": ["a", "b"],
    }


def test_list_slice() -> None:
    df = pl.DataFrame(
        {
            "lst": [[1, 2, 3, 4], [10, 2, 1]],
            "offset": [1, 2],
            "len": [3, 2],
        }
    )

    assert df.select([pl.col("lst").arr.slice("offset", "len")]).to_dict(False) == {
        "lst": [[2, 3, 4], [1]]
    }
    assert df.select([pl.col("lst").arr.slice("offset", 1)]).to_dict(False) == {
        "lst": [[2], [1]]
    }
    assert df.select([pl.col("lst").arr.slice(-2, "len")]).to_dict(False) == {
        "lst": [[3, 4], [2, 1]]
    }


@typing.no_type_check
def test_list_sliced_get_5186() -> None:
    n = 30
    df = pl.from_dict(
        {
            "ind": pl.arange(0, n, eager=True),
            "inds": np.stack([np.arange(n), -np.arange(n)], axis=-1),
        }
    )

    assert df.select(
        [
            "ind",
            pl.col("inds").arr.first().alias("first_element"),
            pl.col("inds").arr.last().alias("last_element"),
        ]
    )[10:20].frame_equal(
        df[10:20].select(
            [
                "ind",
                pl.col("inds").arr.first().alias("first_element"),
                pl.col("inds").arr.last().alias("last_element"),
            ]
        )
    )


def test_empty_eval_dtype_5546() -> None:
    df = pl.DataFrame([{"a": [{"name": 1}, {"name": 2}]}])

    dtype = df.dtypes[0]

    assert (
        df.limit(0).with_column(
            pl.col("a")
            .arr.eval(pl.element().filter(pl.first().struct.field("name") == 1))
            .alias("a_filtered")
        )
    ).dtypes == [dtype, dtype]


def test_fast_explode_flag() -> None:
    df1 = pl.DataFrame({"values": [[[1, 2]]]})
    assert df1.clone().vstack(df1)["values"].flags["FAST_EXPLODE"]


def test_list_amortized_apply_explode_5812() -> None:
    s = pl.Series([None, [1, 3], [0, -3], [1, 2, 2]])
    assert s.arr.sum().to_list() == [None, 4, -3, 5]
    assert s.arr.min().to_list() == [None, 1, -3, 1]
    assert s.arr.max().to_list() == [None, 3, 0, 2]
    assert s.arr.arg_min().to_list() == [None, 0, 1, 0]
    assert s.arr.arg_max().to_list() == [None, 1, 0, 1]


def test_list_slice_5866() -> None:
    vals = [[1, 2, 3, 4], [10, 2, 1]]
    s = pl.Series("a", vals)
    assert s.arr.slice(1).to_list() == [[2, 3, 4], [2, 1]]


def test_list_take() -> None:
    s = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8]])
    # mypy: we make it work, but idomatic is `arr.get`.
    assert s.arr.take(0).to_list() == [[1], [4], [6]]  # type: ignore[arg-type]
    assert s.arr.take([0, 1]).to_list() == [[1, 2], [4, 5], [6, 7]]

    assert s.arr.take([-1, 1]).to_list() == [[3, 2], [5, 5], [8, 7]]

    # use another list to make sure negative indices are respected
    taker = pl.Series([[-1, 1], [-1, 1], [-1, -2]])
    assert s.arr.take(taker).to_list() == [[3, 2], [5, 5], [8, 7]]
    with pytest.raises(pl.ComputeError, match=r"Take indices are out of bounds"):
        s.arr.take([1, 2])
    s = pl.Series(
        [["A", "B", "C"], ["A"], ["B"], ["1", "2"], ["e"]],
    )

    assert s.arr.take([0, 2], null_on_oob=True).to_list() == [
        ["A", "C"],
        ["A", None],
        ["B", None],
        ["1", None],
        ["e", None],
    ]
    assert s.arr.take([0, 1, 2], null_on_oob=True).to_list() == [
        ["A", "B", "C"],
        ["A", None, None],
        ["B", None, None],
        ["1", "2", None],
        ["e", None, None],
    ]
    s = pl.Series([[42, 1, 2], [5, 6, 7]])

    with pytest.raises(pl.ComputeError, match=r"Take indices are out of bounds"):
        s.arr.take([[0, 1, 2, 3], [0, 1, 2, 3]])

    assert s.arr.take([0, 1, 2, 3], null_on_oob=True).to_list() == [
        [42, 1, 2, None],
        [5, 6, 7, None],
    ]


def test_fast_explode_on_list_struct_6208() -> None:
    data = [
        {
            "label": "l",
            "tag": "t",
            "ref": 1,
            "parent": [{"ref": 1, "tag": "t", "ratio": 62.3}],
        },
        {"label": "l", "tag": "t", "ref": 1, "parent": None},
    ]

    df = pl.DataFrame(
        data,
        columns={
            "label": pl.Utf8,
            "tag": pl.Utf8,
            "ref": pl.Int64,
            "parents": pl.List(
                pl.Struct({"ref": pl.Int64, "tag": pl.Utf8, "ratio": pl.Float64})
            ),
        },
    )

    assert not df["parents"].flags["FAST_EXPLODE"]
    assert df.explode("parents").to_dict(False) == {
        "label": ["l", "l"],
        "tag": ["t", "t"],
        "ref": [1, 1],
        "parents": [
            {"ref": 1, "tag": "t", "ratio": 62.3},
            {"ref": None, "tag": None, "ratio": None},
        ],
    }
