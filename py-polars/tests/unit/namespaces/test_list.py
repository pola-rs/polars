from __future__ import annotations

import typing
from datetime import date, datetime

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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
    # exact on oob boundary
    df = pl.DataFrame(
        {
            "index": [3, 3, 3],
            "lists": [[3, 4, 5], [4, 5, 6], [7, 8, 9, 4]],
        }
    )

    assert df.select(pl.col("lists").arr.get(3)).to_dict(False) == {
        "lists": [None, None, 4]
    }
    assert df.select(pl.col("lists").arr.get(pl.col("index"))).to_dict(False) == {
        "lists": [None, None, 4]
    }


def test_contains() -> None:
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    out = a.arr.contains(2)
    expected = pl.Series("a", [True, True, False])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).arr.contains(2)).to_series()
    assert_series_equal(out, expected)


def test_list_concat() -> None:
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
        {"cars_first": [1, 2, 4, None], "cars_literal": [2, 1, 3, 3]},
        schema_overrides={"cars_literal": pl.Int32},  # Literals default to Int32
    )
    assert_frame_equal(out, expected)


def test_list_argminmax() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [0, 2], dtype=pl.UInt32)
    assert_series_equal(s.arr.arg_min(), expected)
    expected = pl.Series("a", [1, 0], dtype=pl.UInt32)
    assert_series_equal(s.arr.arg_max(), expected)


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


def test_list_eval_dtype_inference() -> None:
    grades = pl.DataFrame(
        {
            "student": ["bas", "laura", "tim", "jenny"],
            "arithmetic": [10, 5, 6, 8],
            "biology": [4, 6, 2, 7],
            "geography": [8, 4, 9, 7],
        }
    )

    rank_pct = pl.col("").rank(descending=True) / pl.col("").count().cast(pl.UInt16)

    # the .arr.first() would fail if .arr.eval did not correctly infer the output type
    assert grades.with_columns(
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


def test_list_ternary_concat() -> None:
    df = pl.DataFrame(
        {
            "list1": [["123", "456"], None],
            "list2": [["789"], ["zzz"]],
        }
    )

    assert df.with_columns(
        pl.when(pl.col("list1").is_null())
        .then(pl.col("list1").arr.concat(pl.col("list2")))
        .otherwise(pl.col("list2"))
        .alias("result")
    ).to_dict(False) == {
        "list1": [["123", "456"], None],
        "list2": [["789"], ["zzz"]],
        "result": [["789"], None],
    }

    assert df.with_columns(
        pl.when(pl.col("list1").is_null())
        .then(pl.col("list2"))
        .otherwise(pl.col("list1").arr.concat(pl.col("list2")))
        .alias("result")
    ).to_dict(False) == {
        "list1": [["123", "456"], None],
        "list2": [["789"], ["zzz"]],
        "result": [["123", "456", "789"], ["zzz"]],
    }


def test_arr_contains_categorical() -> None:
    df = pl.DataFrame(
        {"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]}
    ).lazy()
    df = df.with_columns(pl.col("str").cast(pl.Categorical))
    df_groups = df.groupby("group").agg([pl.col("str").alias("str_list")])
    assert df_groups.filter(pl.col("str_list").arr.contains("C")).collect().to_dict(
        False
    ) == {"group": [2], "str_list": [["A", "C"]]}


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
    # https://github.com/pola-rs/polars/issues/5186
    n = 30
    df = pl.from_dict(
        {
            "ind": pl.arange(0, n, eager=True),
            "inds": np.stack([np.arange(n), -np.arange(n)], axis=-1),
        }
    )

    exprs = [
        "ind",
        pl.col("inds").arr.first().alias("first_element"),
        pl.col("inds").arr.last().alias("last_element"),
    ]
    out1 = df.select(exprs)[10:20]
    out2 = df[10:20].select(exprs)
    assert_frame_equal(out1, out2)


def test_empty_eval_dtype_5546() -> None:
    # https://github.com/pola-rs/polars/issues/5546
    df = pl.DataFrame([{"a": [{"name": 1}, {"name": 2}]}])

    dtype = df.dtypes[0]

    assert (
        df.limit(0).with_columns(
            pl.col("a")
            .arr.eval(pl.element().filter(pl.first().struct.field("name") == 1))
            .alias("a_filtered")
        )
    ).dtypes == [dtype, dtype]


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
    with pytest.raises(pl.ComputeError, match=r"take indices are out of bounds"):
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

    with pytest.raises(pl.ComputeError, match=r"take indices are out of bounds"):
        s.arr.take([[0, 1, 2, 3], [0, 1, 2, 3]])

    assert s.arr.take([0, 1, 2, 3], null_on_oob=True).to_list() == [
        [42, 1, 2, None],
        [5, 6, 7, None],
    ]


def test_list_eval_all_null() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, None, None]}).with_columns(
        pl.col("bar").cast(pl.List(pl.Utf8))
    )

    assert df.select(pl.col("bar").arr.eval(pl.element())).to_dict(False) == {
        "bar": [None, None, None]
    }


def test_list_function_group_awareness() -> None:
    df = pl.DataFrame(
        {
            "a": [100, 103, 105, 106, 105, 104, 103, 106, 100, 102],
            "group": [0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
        }
    )

    assert df.groupby("group").agg(
        [
            pl.col("a").list().arr.get(0).alias("get"),
            pl.col("a").list().arr.take([0]).alias("take"),
            pl.col("a").list().arr.slice(0, 3).alias("slice"),
        ]
    ).sort("group").to_dict(False) == {
        "group": [0, 1, 2],
        "get": [100, 105, 100],
        "take": [[100], [105], [100]],
        "slice": [[100, 103], [105, 106, 105], [100, 102]],
    }


def test_list_get_logical_types() -> None:
    df = pl.DataFrame(
        {
            "date_col": [[datetime(2023, 2, 1).date(), datetime(2023, 2, 2).date()]],
            "datetime_col": [[datetime(2023, 2, 1), datetime(2023, 2, 2)]],
        }
    )

    assert df.select(pl.all().arr.get(1).suffix("_element_1")).to_dict(False) == {
        "date_col_element_1": [date(2023, 2, 2)],
        "datetime_col_element_1": [datetime(2023, 2, 2, 0, 0)],
    }


def test_list_take_logical_type() -> None:
    df = pl.DataFrame(
        {"foo": [["foo", "foo", "bar"]], "bar": [[5.0, 10.0, 12.0]]}
    ).with_columns(pl.col("foo").cast(pl.List(pl.Categorical)))

    df = pl.concat([df, df], rechunk=False)
    assert df.n_chunks() == 2
    assert df.select(pl.all().take([0, 1])).to_dict(False) == {
        "foo": [["foo", "foo", "bar"], ["foo", "foo", "bar"]],
        "bar": [[5.0, 10.0, 12.0], [5.0, 10.0, 12.0]],
    }


def test_list_unique() -> None:
    assert (
        pl.Series([[1, 1, 2, 2, 3], [3, 3, 3, 2, 1, 2]])
        .arr.unique(maintain_order=True)
        .series_equal(pl.Series([[1, 2, 3], [3, 2, 1]]))
    )
