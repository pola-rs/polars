# flake8: noqa: W191,E101
import sys
import typing as tp
from builtins import range
from datetime import datetime
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl


def test_version() -> None:
    pl.__version__


def test_init_empty() -> None:
    # Empty initialization
    df1 = pl.DataFrame()
    assert df1.shape == (0, 0)


def test_init_only_columns() -> None:
    df = pl.DataFrame(columns=["a", "b", "c"])
    truth = pl.DataFrame({"a": [], "b": [], "c": []})
    assert df.shape == (0, 3)
    assert df.frame_equal(truth, null_equal=True)


def test_init_dict() -> None:
    # Empty dictionary
    df = pl.DataFrame({})
    assert df.shape == (0, 0)

    # Mixed dtypes
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.shape == (3, 2)
    assert df.columns == ["a", "b"]

    # Values contained in tuples
    df = pl.DataFrame({"a": (1, 2, 3), "b": [1.0, 2.0, 3.0]})
    assert df.shape == (3, 2)

    # Overriding dict column names
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, columns=["c", "d"])
    assert df.columns == ["c", "d"]


def test_init_ndarray() -> None:
    # Empty array
    df = pl.DataFrame(np.array([]))
    assert df.frame_equal(pl.DataFrame())

    # 1D array
    df = pl.DataFrame(np.array([1, 2, 3]), columns=["a"])
    truth = pl.DataFrame({"a": [1, 2, 3]})
    assert df.frame_equal(truth)

    # 2D array - default to column orientation
    df = pl.DataFrame(np.array([[1, 2], [3, 4]]), orient="column")
    truth = pl.DataFrame({"column_0": [1, 2], "column_1": [3, 4]})
    assert df.frame_equal(truth)

    df = pl.DataFrame(np.array([[1, 2], [3, 4]]), orient="row")
    truth = pl.DataFrame({"column_0": [1, 3], "column_1": [2, 4]})
    assert df.frame_equal(truth)

    # TODO: Uncomment tests below when removing deprecation warning
    # # 2D array - default to column orientation
    # df = pl.DataFrame(np.array([[1, 2], [3, 4]]))
    # truth = pl.DataFrame({"column_0": [1, 2], "column_1": [3, 4]})
    # assert df.frame_equal(truth)

    # # 2D array - row orientation inferred
    # df = pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b", "c"])
    # truth = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    # assert df.frame_equal(truth)

    # # 2D array - column orientation inferred
    # df = pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b"])
    # truth = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # assert df.frame_equal(truth)

    # 2D array - orientation conflicts with columns
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b"], orient="row")

    # 3D array
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.random.randn(2, 2, 2))


# TODO: Remove this test case when removing deprecated behaviour
def test_init_ndarray_deprecated() -> None:
    with pytest.deprecated_call():
        # 2D array - default to row orientation
        df = pl.DataFrame(np.array([[1, 2], [3, 4]]))
        truth = pl.DataFrame({"column_0": [1, 3], "column_1": [2, 4]})
        assert df.frame_equal(truth)


def test_init_arrow() -> None:
    # Handle unnamed column
    df = pl.DataFrame(pa.table({"a": [1, 2], None: [3, 4]}))
    truth = pl.DataFrame({"a": [1, 2], "None": [3, 4]})
    assert df.frame_equal(truth)

    # Rename columns
    df = pl.DataFrame(pa.table({"a": [1, 2], "b": [3, 4]}), columns=["c", "d"])
    truth = pl.DataFrame({"c": [1, 2], "d": [3, 4]})
    assert df.frame_equal(truth)

    # Bad columns argument
    with pytest.raises(ValueError):
        pl.DataFrame(
            pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), columns=["c", "d", "e"]
        )


def test_init_series() -> None:
    # List of Series
    df = pl.DataFrame([pl.Series("a", [1, 2, 3]), pl.Series("b", [4, 5, 6])])
    truth = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.frame_equal(truth)

    # Tuple of Series
    df = pl.DataFrame((pl.Series("a", (1, 2, 3)), pl.Series("b", (4, 5, 6))))
    assert df.frame_equal(truth)

    # List of unnamed Series
    df = pl.DataFrame([pl.Series([1, 2, 3]), pl.Series([4, 5, 6])])
    truth = pl.DataFrame(
        [pl.Series("column_0", [1, 2, 3]), pl.Series("column_1", [4, 5, 6])]
    )
    assert df.frame_equal(truth)

    # Single Series
    df = pl.DataFrame(pl.Series("a", [1, 2, 3]))
    truth = pl.DataFrame({"a": [1, 2, 3]})
    assert df.frame_equal(truth)


def test_init_seq_of_seq() -> None:
    # List of lists
    df = pl.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    truth = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert df.frame_equal(truth)

    # Tuple of tuples, default to column orientation
    df = pl.DataFrame(((1, 2, 3), (4, 5, 6)))
    truth = pl.DataFrame({"column_0": [1, 2, 3], "column_1": [4, 5, 6]})
    assert df.frame_equal(truth)

    # Row orientation
    df = pl.DataFrame(((1, 2), (3, 4)), columns=("a", "b"), orient="row")
    truth = pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert df.frame_equal(truth)


def test_init_1d_sequence() -> None:
    # Empty list
    df = pl.DataFrame([])
    assert df.frame_equal(pl.DataFrame())

    # List of strings
    df = pl.DataFrame(["a", "b", "c"], columns=["hi"])
    truth = pl.DataFrame({"hi": ["a", "b", "c"]})
    assert df.frame_equal(truth)

    # String sequence
    with pytest.raises(ValueError):
        pl.DataFrame("abc")


def test_init_pandas() -> None:
    pandas_df = pd.DataFrame([[1, 2], [3, 4]], columns=[1, 2])

    # pandas is available; integer column names
    with patch("polars.internals.frame._PANDAS_AVAILABLE", True):
        df = pl.DataFrame(pandas_df)
        truth = pl.DataFrame({"1": [1, 3], "2": [2, 4]})
        assert df.frame_equal(truth)

    # pandas is not available
    with patch("polars.internals.frame._PANDAS_AVAILABLE", False):
        with pytest.raises(ValueError):
            pl.DataFrame(pandas_df)


def test_init_errors() -> None:
    # Length mismatch
    with pytest.raises(RuntimeError):
        pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0, 4.0]})

    # Columns don't match data dimensions
    with pytest.raises(RuntimeError):
        pl.DataFrame([[1, 2], [3, 4]], columns=["a", "b", "c"])

    # Unmatched input
    with pytest.raises(ValueError):
        pl.DataFrame(0)


def test_init_records() -> None:
    dicts = [
        {"a": 1, "b": 2},
        {"b": 1, "a": 2},
        {"a": 1, "b": 2},
    ]
    df = pl.DataFrame(dicts)
    expected = pl.DataFrame({"a": [1, 2, 1], "b": [2, 1, 2]})
    assert df.frame_equal(expected)
    assert df.to_dicts() == dicts

    df_cd = pl.DataFrame(dicts, columns=["c", "d"])
    expected = pl.DataFrame({"c": [1, 2, 1], "d": [2, 1, 2]})
    assert df_cd.frame_equal(expected)


def test_selection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    # get_column by name
    assert df.get_column("a").to_list() == [1, 2, 3]

    # select columns by mask
    assert df[:2, [True, False, False]].shape == (2, 1)
    assert df[:2, pl.Series([True, False, False])].shape == (2, 1)

    # column selection by string(s) in first dimension
    assert df["a"].to_list() == [1, 2, 3]
    assert df["b"].to_list() == [1.0, 2.0, 3.0]
    assert df["c"].to_list() == ["a", "b", "c"]

    # row selection by integers(s) in first dimension
    assert df[0].frame_equal(pl.DataFrame({"a": [1], "b": [1.0], "c": ["a"]}))
    assert df[-1].frame_equal(pl.DataFrame({"a": [3], "b": [3.0], "c": ["c"]}))

    # row, column selection when using two dimensions
    assert df[:, 0] == [1, 2, 3]
    assert df[:, 1] == [1.0, 2.0, 3.0]
    assert df[:2, 2] == ["a", "b"]

    assert df[[1, 2]].frame_equal(
        pl.DataFrame({"a": [2, 3], "b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert df[[-1, -2]].frame_equal(
        pl.DataFrame({"a": [3, 2], "b": [3.0, 2.0], "c": ["c", "b"]})
    )

    assert df[[True, False, True]].frame_equal(
        pl.DataFrame({"a": [1, 3], "b": [1.0, 3.0], "c": ["a", "c"]})
    )
    assert df[["a", "b"]].columns == ["a", "b"]
    assert df[[1, 2], [1, 2]].frame_equal(
        pl.DataFrame({"b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert df[1, 2] == "b"
    assert df[1, 1] == 2.0
    assert df[2, 0] == 3

    assert df[[True, False, True], "b"].shape == (2, 1)
    assert df[[True, False, False], ["a", "b"]].shape == (1, 2)

    assert df[[0, 1], "b"].shape == (2, 1)
    assert df[[2], ["a", "b"]].shape == (1, 2)
    assert df.select_at_idx(0).name == "a"
    assert (df.a == df["a"]).sum() == 3
    assert (df.c == df["a"]).sum() == 0
    assert df[:, "a":"b"].shape == (3, 2)  # type: ignore
    assert df[:, "a":"c"].columns == ["a", "b", "c"]  # type: ignore
    expect = pl.DataFrame({"c": ["b"]})
    assert df[1, [2]].frame_equal(expect)
    expect = pl.DataFrame({"b": [1.0, 3.0]})
    assert df[[0, 2], [1]].frame_equal(expect)
    assert df[0, "c"] == "a"
    assert df[1, "c"] == "b"
    assert df[2, "c"] == "c"
    assert df[0, "a"] == 1

    # more slicing
    expect = pl.DataFrame({"a": [3, 2, 1], "b": [3.0, 2.0, 1.0], "c": ["c", "b", "a"]})
    assert df[::-1].frame_equal(expect)
    expect = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["a", "b"]})
    assert df[:-1].frame_equal(expect)

    expect = pl.DataFrame({"a": [1, 3], "b": [1.0, 3.0], "c": ["a", "c"]})
    assert df[::2].frame_equal(expect)


def test_from_arrow() -> None:
    tbl = pa.table(
        {
            "a": pa.array([1, 2], pa.timestamp("s")),
            "b": pa.array([1, 2], pa.timestamp("ms")),
            "c": pa.array([1, 2], pa.timestamp("us")),
            "d": pa.array([1, 2], pa.timestamp("ns")),
            "decimal1": pa.array([1, 2], pa.decimal128(2, 1)),
        }
    )
    assert pl.from_arrow(tbl).shape == (2, 5)


def test_sort() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    df.sort("a", in_place=True)
    assert df.frame_equal(pl.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]}))


def test_replace() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    s = pl.Series("c", [True, False, True])
    df.replace("a", s)
    assert df.frame_equal(pl.DataFrame({"a": [True, False, True], "b": [1, 2, 3]}))


def test_assignment() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [2, 3, 4]})
    df["foo"] = df["foo"]
    # make sure that assignment does not change column order
    assert df.columns == ["foo", "bar"]
    df[df["foo"] > 1, "foo"] = 9
    assert df["foo"].to_list() == [1, 9, 9]


def test_slice() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df = df.slice(1, 2)
    assert df.frame_equal(pl.DataFrame({"a": [1, 3], "b": ["b", "c"]}))


def test_null_count() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", None]})
    assert df.null_count().shape == (1, 2)


def test_head_tail() -> None:
    df = pl.DataFrame({"a": range(10), "b": range(10)})
    assert df.head(5).height == 5
    assert df.tail(5).height == 5

    assert not df.head(5).frame_equal(df.tail(5))
    # check if it doesn't fail when out of bounds
    assert df.head(100).height == 10
    assert df.tail(100).height == 10


def test_explode() -> None:
    df = pl.DataFrame({"letters": ["c", "a"], "nrs": [[1, 2], [1, 3]]})
    out = df.explode("nrs")
    assert out["letters"].to_list() == ["c", "c", "a", "a"]
    assert out["nrs"].to_list() == [1, 2, 1, 3]


def test_groupby() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    gb_df = df.groupby("a").agg({"b": ["sum", "min"], "c": "count"})
    assert "b_sum" in gb_df.columns
    assert "b_min" in gb_df.columns

    #
    # # TODO: is false because count is u32
    # df.groupby(by="a", select="b", agg="count").frame_equal(
    #     pl.DataFrame({"a": ["a", "b", "c"], "": [2, 3, 1]})
    # )
    assert df.groupby("a").apply(lambda df: df[["c"]].sum()).sort("c")["c"][0] == 1

    assert (
        df.groupby("a")
        .groups()
        .sort("a")["a"]
        .series_equal(pl.Series("a", ["a", "b", "c"]))
    )

    for subdf in df.groupby("a"):  # type: ignore
        # TODO: add __next__() to GroupBy
        if subdf["a"][0] == "b":
            assert subdf.shape == (3, 3)

    assert df.groupby("a").get_group("c").shape == (1, 3)
    assert df.groupby("a").get_group("b").shape == (3, 3)
    assert df.groupby("a").get_group("a").shape == (2, 3)

    # Use lazy API in eager groupby
    assert df.groupby("a").agg([pl.sum("b")]).shape == (3, 2)
    # test if it accepts a single expression
    assert df.groupby("a").agg(pl.sum("b")).shape == (3, 2)

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "a", "b", "b", "b"],
            "c": [None, 1, None, 1, None],
        }
    )

    # check if this query runs and thus column names propagate
    df.groupby("b").agg(pl.col("c").forward_fill()).explode("c")


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

    joined = df_left.join(df_right, left_on="a", right_on="a").sort("a")
    assert joined["b"].series_equal(pl.Series("b", [1, 3, 2, 2]))
    joined = df_left.join(df_right, left_on="a", right_on="a", how="left").sort("a")
    assert joined["c_right"].is_null().sum() == 1
    assert joined["b"].series_equal(pl.Series("b", [1, 3, 2, 2, 4]))
    joined = df_left.join(df_right, left_on="a", right_on="a", how="outer").sort("a")
    assert joined["c_right"].null_count() == 1
    assert joined["c"].null_count() == 1
    assert joined["b"].null_count() == 1
    assert joined["k"].null_count() == 1
    assert joined["a"].null_count() == 0

    df_a = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})
    df_b = pl.DataFrame(
        {"foo": [1, 1, 1], "bar": ["a", "c", "c"], "ham": ["let", "var", "const"]}
    )

    # just check if join on multiple columns runs
    df_a.join(df_b, left_on=["a", "b"], right_on=["foo", "bar"])

    eager_join = df_a.join(df_b, left_on="a", right_on="foo")

    lazy_join = df_a.lazy().join(df_b.lazy(), left_on="a", right_on="foo").collect()
    assert lazy_join.shape == eager_join.shape


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
        [pl.col("date").str.strptime(pl.Date), pl.col("datetime").cast(pl.Datetime)]
    )

    for how in ["left", "inner", "outer"]:
        dfa.join(dfa, on=["a", "b", "date", "datetime"], how=how)
        dfa.join(dfa, on=["date", "datetime"], how=how)
        dfa.join(dfa, on=["date", "datetime", "a"], how=how)
        dfa.join(dfa, on=["date", "a"], how=how)
        dfa.join(dfa, on=["a", "datetime"], how=how)
        dfa.join(dfa, on=["date"], how=how)


def test_hstack() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df.hstack([pl.Series("stacked", [-1, -1, -1])], in_place=True)
    assert df.shape == (3, 3)
    assert df.columns == ["a", "b", "stacked"]


def test_drop() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    df = df.drop("a")
    assert df.shape == (3, 2)
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    s = df.drop_in_place("a")
    assert s.name == "a"


def test_file_buffer() -> None:
    f = BytesIO()
    f.write(b"1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    df = pl.DataFrame.read_csv(f, has_header=False)
    assert df.shape == (2, 6)
    f.seek(0)

    # check if not fails on TryClone and Length impl in file.rs
    with pytest.raises(RuntimeError) as e:
        df.read_parquet(f)
    assert "Invalid Parquet file" in str(e.value)


def test_set() -> None:
    np.random.seed(1)
    df = pl.DataFrame(
        {"foo": np.random.rand(10), "bar": np.arange(10), "ham": ["h"] * 10}
    )
    df["new"] = np.random.rand(10)
    df[df["new"] > 0.5, "new"] = 1

    df = pl.DataFrame({"b": [0, 0]})
    df[["A", "B"]] = [[1, 2], [1, 2]]  # type: ignore
    assert df["A"] == [1, 1]
    assert df["B"] == [2, 2]

    df = pl.DataFrame({"b": [0, 0]})
    df[0, "b"] = 1
    assert df[0, "b"] == 1

    df[0, 0] = 2
    assert df[0, "b"] == 2


def test_melt() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    melted = df.melt(id_vars="A", value_vars=["B", "C"])
    assert melted["value"] == [1, 3, 4, 2, 4, 6]


def test_shift() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    a = df.shift(1)
    b = pl.DataFrame(
        {"A": [None, "a", "b"], "B": [None, 1, 3]},
    )
    assert a.frame_equal(b, null_equal=True)


def test_to_dummies() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    dummies = df.to_dummies()
    assert dummies["A_a"].to_list() == [1, 0, 0]
    assert dummies["A_b"].to_list() == [0, 1, 0]
    assert dummies["A_c"].to_list() == [0, 0, 1]


def test_from_pandas() -> None:
    df = pd.DataFrame(
        {
            "bools": [False, True, False],
            "bools_nulls": [None, True, False],
            "int": [1, 2, 3],
            "int_nulls": [1, None, 3],
            "floats": [1.0, 2.0, 3.0],
            "floats_nulls": [1.0, None, 3.0],
            "strings": ["foo", "bar", "ham"],
            "strings_nulls": ["foo", None, "ham"],
            "strings-cat": ["foo", "bar", "ham"],
        }
    )
    df["strings-cat"] = df["strings-cat"].astype("category")

    out = pl.from_pandas(df)
    assert out.shape == (3, 9)


def test_from_pandas_nan_to_none() -> None:
    from pyarrow import ArrowInvalid

    df = pd.DataFrame(
        {
            "bools_nulls": [None, True, False],
            "int_nulls": [1, None, 3],
            "floats_nulls": [1.0, None, 3.0],
            "strings_nulls": ["foo", None, "ham"],
            "nulls": [None, np.nan, np.nan],
        }
    )
    out_true = pl.from_pandas(df)
    out_false = pl.from_pandas(df, nan_to_none=False)
    df.loc[2, "nulls"] = pd.NA
    assert [val is None for val in out_true["nulls"]]
    assert [np.isnan(val) for val in out_false["nulls"][1:]]
    with pytest.raises(ArrowInvalid, match="Could not convert"):
        pl.from_pandas(df, nan_to_none=False)


def test_custom_groupby() -> None:
    df = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})

    out = (
        df.lazy()
        .groupby("b")
        .agg([pl.col("a").apply(lambda x: x.sum(), return_dtype=pl.Int64)])
        .collect()
    )
    assert out.shape == (3, 2)


def test_multiple_columns_drop() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    out = df.drop(["a", "b"])
    assert out.columns == ["c"]


def test_concat() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    df2 = pl.concat([df, df])
    assert df2.shape == (6, 3)
    assert df2.n_chunks() == 1  # the default is to rechunk

    assert pl.concat([df, df], rechunk=False).n_chunks() == 2

    # check if a remains unchanged
    a = pl.from_records(((1, 2), (1, 2)))
    _ = pl.concat([a, a, a])
    assert a.shape == (2, 2)


def test_arg_where() -> None:
    s = pl.Series([True, False, True, False])
    assert pl.arg_where(s).cast(int).series_equal(pl.Series([0, 2]))


def test_get_dummies() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    res = pl.get_dummies(df)
    expected = pl.DataFrame(
        {"a_1": [1, 0, 0], "a_2": [0, 1, 0], "a_3": [0, 0, 1]}
    ).with_columns(pl.all().cast(pl.UInt8))
    assert res.frame_equal(expected)


def test_to_pandas(df: pl.DataFrame) -> None:
    # pyarrow cannot deal with unsigned dictionary integer yet.
    # pyarrow cannot convert a time64 w/ non-zero nanoseconds
    df = df.drop(["cat", "time"])
    df.to_arrow()
    df.to_pandas()
    # test shifted df
    df.shift(2).to_pandas()
    df = pl.DataFrame({"col": pl.Series([True, False, True])})
    df.shift(2).to_pandas()


def test_from_arrow_table() -> None:
    data = {"a": [1, 2], "b": [1, 2]}
    tbl = pa.table(data)

    df: pl.DataFrame = pl.from_arrow(tbl)  # type: ignore
    df.frame_equal(pl.DataFrame(data))


def test_df_stats(df: pl.DataFrame) -> None:
    df.var()
    df.std()
    df.min()
    df.max()
    df.sum()
    df.mean()
    df.median()
    df.quantile(0.4, "nearest")


def test_df_fold() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    assert df.fold(lambda s1, s2: s1 + s2).series_equal(pl.Series("a", [4.0, 5.0, 9.0]))
    assert df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2)).series_equal(
        pl.Series("a", [1.0, 1.0, 3.0])
    )

    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.fold(lambda s1, s2: s1 + s2)
    out.series_equal(pl.Series("", ["foo11", "bar22", "233"]))

    df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    # just check dispatch. values are tested on rust side.
    assert len(df.sum(axis=1)) == 3
    assert len(df.mean(axis=1)) == 3
    assert len(df.min(axis=1)) == 3
    assert len(df.max(axis=1)) == 3


def test_row_tuple() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    assert df.row(0) == ("foo", 1, 1.0)
    assert df.row(1) == ("bar", 2, 2.0)
    assert df.row(-1) == ("2", 3, 3.0)


def test_read_csv_categorical() -> None:
    f = BytesIO()
    f.write(b"col1,col2,col3,col4,col5,col6\n'foo',2,3,4,5,6\n'bar',8,9,10,11,12")
    f.seek(0)
    df = pl.DataFrame.read_csv(f, has_header=True, dtypes={"col1": pl.Categorical})
    assert df["col1"].dtype == pl.Categorical


def test_df_apply() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.apply(lambda x: len(x), None)
    assert out.sum() == 9


def test_column_names() -> None:
    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5], pa.decimal128(38, 2)),
            "b": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    )
    df: pl.DataFrame = pl.from_arrow(tbl)  # type: ignore
    assert df.columns == ["a", "b"]


def test_lazy_functions() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df[[pl.count("a")]]
    assert out["a"] == 3
    assert pl.count(df["a"]) == 3
    out = df[
        [
            pl.var("b").alias("1"),
            pl.std("b").alias("2"),
            pl.max("b").alias("3"),
            pl.min("b").alias("4"),
            pl.sum("b").alias("5"),
            pl.mean("b").alias("6"),
            pl.median("b").alias("7"),
            pl.n_unique("b").alias("8"),
            pl.first("b").alias("9"),
            pl.last("b").alias("10"),
        ]
    ]
    expected = 1.0
    assert np.isclose(out.select_at_idx(0), expected)
    assert np.isclose(pl.var(df["b"]), expected)
    expected = 1.0
    assert np.isclose(out.select_at_idx(1), expected)
    assert np.isclose(pl.std(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(2), expected)
    assert np.isclose(pl.max(df["b"]), expected)
    expected = 1
    assert np.isclose(out.select_at_idx(3), expected)
    assert np.isclose(pl.min(df["b"]), expected)
    expected = 6
    assert np.isclose(out.select_at_idx(4), expected)
    assert np.isclose(pl.sum(df["b"]), expected)
    expected = 2
    assert np.isclose(out.select_at_idx(5), expected)
    assert np.isclose(pl.mean(df["b"]), expected)
    expected = 2
    assert np.isclose(out.select_at_idx(6), expected)
    assert np.isclose(pl.median(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(7), expected)
    assert np.isclose(pl.n_unique(df["b"]), expected)
    expected = 1
    assert np.isclose(out.select_at_idx(8), expected)
    assert np.isclose(pl.first(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(9), expected)
    assert np.isclose(pl.last(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(9), expected)
    assert np.isclose(pl.last(df["b"]), expected)


def test_multiple_column_sort() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [2, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.sort([pl.col("b"), pl.col("c").reverse()])
    assert out["c"] == [2, 3, 1]
    assert out["b"] == [2, 2, 3]

    df = pl.DataFrame({"a": np.arange(1, 4), "b": ["a", "a", "b"]})

    df.sort("a", reverse=True).frame_equal(
        pl.DataFrame({"a": [3, 2, 1], "b": ["b", "a", "a"]})
    )

    df.sort("b", reverse=True).frame_equal(
        pl.DataFrame({"a": [3, 1, 2], "b": ["b", "a", "a"]})
    )

    df.sort(["b", "a"], reverse=[False, True]).frame_equal(
        pl.DataFrame({"a": [2, 1, 3], "b": ["a", "a", "b"]})
    )


def test_describe() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, 6],
            "c": [True, False, True],
            "d": ["a", "b", "c"],
        }
    )
    assert df.describe().shape != df.shape
    assert set(df.describe().select_at_idx(2)) == {1.0, 4.0, 5.0, 6.0}


def test_string_cache_eager_lazy() -> None:
    # tests if the global string cache is really global and not interfered by the lazy execution.
    # first the global settings was thread-local and this breaks with the parallel execution of lazy
    with pl.StringCache():
        df1 = pl.DataFrame(
            {"region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"]}
        ).select([pl.col("region_ids").cast(pl.Categorical)])
        df2 = pl.DataFrame(
            {"seq_name": ["reg4", "reg2", "reg1"], "score": [3.0, 1.0, 2.0]}
        ).select([pl.col("seq_name").cast(pl.Categorical), pl.col("score")])

    expected = pl.DataFrame(
        {
            "region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"],
            "score": [2.0, 1.0, None, 3.0, None],
        }
    ).with_column(pl.col("region_ids").cast(pl.Categorical))

    assert df1.join(
        df2, left_on="region_ids", right_on="seq_name", how="left"
    ).frame_equal(expected, null_equal=True)


def test_assign() -> None:
    # check if can assign in case of a single column
    df = pl.DataFrame({"a": [1, 2, 3]})
    # test if we can assign in case of single column
    df["a"] = df["a"] * 2
    assert df["a"] == [2, 4, 6]


def test_to_numpy() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.to_numpy().shape == (3, 2)


def test_argsort_by(df: pl.DataFrame) -> None:
    a = df[pl.argsort_by(["int_nulls", "floats"], reverse=[False, True])]["int_nulls"]
    assert a == [1, 0, 3]

    a = df[pl.argsort_by(["int_nulls", "floats"], reverse=False)]["int_nulls"]
    assert a == [1, 0, 2]


def test_literal_series() -> None:
    df = pl.DataFrame(
        {
            "a": np.array([21.7, 21.8, 21], dtype=np.float32),
            "b": np.array([1, 3, 2], dtype=np.int64),
            "c": ["reg1", "reg2", "reg3"],
        }
    )
    out = (
        df.lazy()
        .with_column(pl.Series("e", [2, 1, 3]))  # type: ignore
        .with_column(pl.col("e").cast(pl.Float32))
        .collect()
    )
    assert out["e"] == [2, 1, 3]


def test_to_html(df: pl.DataFrame) -> None:
    # check if it does not panic/ error
    df._repr_html_()


def test_rows() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    assert df.rows() == [(1, 1), (2, 2)]


def test_rename(df: pl.DataFrame) -> None:
    out = df.rename({"strings": "bars", "int": "foos"})
    # check if wel can select these new columns
    _ = out[["foos", "bars"]]


def test_to_json(df: pl.DataFrame) -> None:
    # text based conversion loses time info
    df = df.select(pl.all().exclude(["cat", "time"]))
    s: str = df.to_json(to_string=True)  # type: ignore
    # TODO add overload on to_json()
    out = pl.read_json(s)
    assert df.frame_equal(out, null_equal=True)


def test_from_rows() -> None:
    df = pl.from_records([[1, 2, "foo"], [2, 3, "bar"]], orient="row")
    assert df.frame_equal(
        pl.DataFrame(
            {"column_0": [1, 2], "column_1": [2, 3], "column_2": ["foo", "bar"]}
        )
    )

    df = pl.from_records(
        [[1, datetime.fromtimestamp(100)], [2, datetime.fromtimestamp(2398754908)]],
        orient="row",
    )
    assert df.dtypes == [pl.Int64, pl.Datetime]


def test_repeat_by() -> None:
    df = pl.DataFrame({"name": ["foo", "bar"], "n": [2, 3]})

    out = df[pl.col("n").repeat_by("n")]
    s = out["n"]
    assert s[0] == [2, 2]
    assert s[1] == [3, 3, 3]


def test_join_dates() -> None:
    date_times = pd.date_range(
        "2021-06-24 00:00:00", "2021-06-24 10:00:00", freq="1H", closed="left"
    )
    dts = (
        pl.from_pandas(date_times)
        .apply(lambda x: x + np.random.randint(1_000 * 60, 60_000 * 60))  # type: ignore
        .cast(pl.Datetime)
    )

    # some df with sensor id, (randomish) datetime and some value
    df = pl.DataFrame(
        {
            "sensor": ["a"] * 5 + ["b"] * 5,
            "datetime": dts,
            "value": [2, 3, 4, 1, 2, 3, 5, 1, 2, 3],
        }
    )
    df.join(df, on="datetime")


def test_asof_cross_join() -> None:
    left = pl.DataFrame({"a": [-10, 5, 10], "left_val": ["a", "b", "c"]})
    right = pl.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

    # only test dispatch of asof join
    out = left.join(right, on="a", how="asof")
    assert out.shape == (3, 3)

    left.lazy().join(right.lazy(), on="a", how="asof").collect()
    assert out.shape == (3, 3)

    # only test dispatch of cross join
    out = left.join(right, how="cross")
    assert out.shape == (15, 4)

    left.lazy().join(right.lazy(), how="cross").collect()
    assert out.shape == (15, 4)


def test_str_concat() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4],
            "name": ["ham", "spam", "foo", None],
        }
    )
    out = df.with_column((pl.lit("Dr. ") + pl.col("name")).alias("graduated_name"))
    assert out["graduated_name"][0] == "Dr. ham"
    assert out["graduated_name"][1] == "Dr. spam"


def dot_product() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})

    assert df["a"].dot(df["b"]) == 20
    assert df[[pl.col("a").dot("b")]][0, "a"] == 20


def test_hash_rows() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})
    assert df.hash_rows().dtype == pl.UInt64
    assert df["a"].hash().dtype == pl.UInt64
    assert df[[pl.col("a").hash().alias("foo")]]["foo"].dtype == pl.UInt64


def test_create_df_from_object() -> None:
    class Foo:
        def __init__(self) -> None:
            pass

    df = pl.DataFrame({"a": [Foo(), Foo()]})
    assert df["a"].dtype == pl.Object


def test_hashing_on_python_objects() -> None:
    # see if we can do a groupby, drop_duplicates on a DataFrame with objects.
    # this requires that the hashing and aggregations are done on python objects

    df = pl.DataFrame({"a": [1, 1, 3, 4], "b": [1, 1, 2, 2]})
    df = df.with_column(pl.col("a").apply(lambda x: datetime(2021, 1, 1)).alias("foo"))
    assert df.groupby(["foo"]).first().shape == (1, 3)
    assert df.drop_duplicates().shape == (3, 3)


def test_drop_duplicates_unit_rows() -> None:
    # simply test if we don't panic.
    pl.DataFrame({"a": [1], "b": [None]}).drop_duplicates(subset="a")


def test_panic() -> None:
    # may contain some tests that yielded a panic in polars or arrow
    # https://github.com/pola-rs/polars/issues/1110
    a = pl.DataFrame(
        {
            "col1": ["a"] * 500 + ["b"] * 500,
        }
    )
    a.filter(pl.col("col1") != "b")


def test_h_agg() -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})

    assert df.sum(axis=1, null_strategy="ignore").to_list() == [2, 2, 6]
    assert df.sum(axis=1, null_strategy="propagate").to_list() == [2, None, 6]
    assert df.mean(axis=1, null_strategy="propagate")[1] is None


def test_slicing() -> None:
    # https://github.com/pola-rs/polars/issues/1322
    n = 20

    df = pl.DataFrame(
        {
            "d": ["u", "u", "d", "c", "c", "d", "d"] * n,
            "v1": [None, "help", None, None, None, None, None] * n,
        }
    )

    assert (df.filter(pl.col("d") != "d").select([pl.col("v1").unique()])).shape == (
        2,
        1,
    )


def test_apply_list_return() -> None:
    df = pl.DataFrame({"start": [1, 2], "end": [3, 5]})
    out = df.apply(lambda r: pl.Series(range(r[0], r[1] + 1)))  # type: ignore
    assert out.to_list() == [[1, 2, 3], [2, 3, 4, 5]]


def test_groupby_cat_list() -> None:  # noqa: W191,E101
    grouped = (
        pl.DataFrame(
            [
                pl.Series("str_column", ["a", "b", "b", "a", "b"]),
                pl.Series("int_column", [1, 1, 2, 2, 3]),
            ]
        )
        .with_column(pl.col("str_column").cast(pl.Categorical).alias("cat_column"))
        .groupby("int_column", maintain_order=True)
        .agg([pl.col("cat_column")])["cat_column"]
    )

    out = grouped.explode()
    assert out.dtype == pl.Categorical
    assert out[0] == "a"

    # test if we can also correctly fmt the categorical in list
    assert (
        str(grouped)
        == """shape: (3,)
Series: 'cat_column' [list]
[
	["a", "b"]
	["b", "a"]
	["b"]
]"""
    )


def test_asof_join() -> None:
    fmt = "%F %T%.3f"
    dates = """2016-05-25 13:30:00.023
2016-05-25 13:30:00.023
2016-05-25 13:30:00.030
2016-05-25 13:30:00.041
2016-05-25 13:30:00.048
2016-05-25 13:30:00.049
2016-05-25 13:30:00.072
2016-05-25 13:30:00.075""".split(
        "\n"
    )

    ticker = """GOOG
MSFT
MSFT
MSFT
GOOG
AAPL
GOOG
MSFT""".split(
        "\n"
    )

    quotes = pl.DataFrame(
        {
            "dates": pl.Series(dates).str.strptime(pl.Datetime, fmt=fmt),
            "ticker": ticker,
            "bid": [720.5, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
        }
    )

    dates = """2016-05-25 13:30:00.023
2016-05-25 13:30:00.038
2016-05-25 13:30:00.048
2016-05-25 13:30:00.048
2016-05-25 13:30:00.048""".split(
        "\n"
    )

    ticker = """MSFT
MSFT
GOOG
GOOG
AAPL""".split(
        "\n"
    )

    trades = pl.DataFrame(
        {
            "dates": pl.Series(dates).str.strptime(pl.Datetime, fmt=fmt),
            "ticker": ticker,
            "bid": [51.95, 51.95, 720.77, 720.92, 98.0],
        }
    )

    out = trades.join(quotes, on="dates", how="asof")
    assert out.columns == ["dates", "ticker", "bid", "ticker_right", "bid_right"]
    assert (out["dates"].cast(int) // 1000_000).to_list() == [
        1464183000023,
        1464183000038,
        1464183000048,
        1464183000048,
        1464183000048,
    ]
    out = trades.join(quotes, on="dates", how="asof", asof_by="ticker")
    assert out["bid_right"].to_list() == [51.95, 51.97, 720.5, 720.5, None]

    out = quotes.join(trades, on="dates", asof_by="ticker", how="asof")
    assert out["bid_right"].to_list() == [
        None,
        51.95,
        51.95,
        51.95,
        720.92,
        98.0,
        720.92,
        51.95,
    ]


def test_groupby_agg_n_unique_floats() -> None:
    # tests proper dispatch
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    for dtype in [pl.Float32, pl.Float64]:
        out = df.groupby("a", maintain_order=True).agg(
            [pl.col("b").cast(dtype).n_unique()]
        )
        assert out["b_n_unique"].to_list() == [2, 1]


def test_select_by_dtype(df: pl.DataFrame) -> None:
    out = df.select(pl.col(pl.Utf8))
    assert out.columns == ["strings", "strings_nulls"]
    out = df.select(pl.col([pl.Utf8, pl.Boolean]))
    assert out.columns == ["strings", "strings_nulls", "bools", "bools_nulls"]


def test_with_row_count() -> None:
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    out = df.with_row_count()
    assert out["row_nr"].to_list() == [0, 1, 2]

    out = df.lazy().with_row_count().collect()
    assert out["row_nr"].to_list() == [0, 1, 2]


def test_filter_with_all_expansion() -> None:
    df = pl.DataFrame(
        {
            "b": [1, 2, None],
            "c": [1, 2, None],
            "a": [None, None, None],
        }
    )
    out = df.filter(~pl.fold(True, lambda acc, s: acc & s.is_null(), pl.all()))  # type: ignore
    assert out.shape == (2, 3)


def test_diag_concat() -> None:
    a = pl.DataFrame({"a": [1, 2]})
    b = pl.DataFrame({"b": ["a", "b"], "c": [1, 2]})
    c = pl.DataFrame({"a": [5, 7], "c": [1, 2], "d": [1, 2]})

    out = pl.concat([a, b, c], how="diagonal")
    expected = pl.DataFrame(
        {
            "a": [1, 2, None, None, 5, 7],
            "b": [None, None, "a", "b", None, None],
            "c": [None, None, 1, 2, 1, 2],
            "d": [None, None, None, None, 1, 2],
        }
    )

    assert out.frame_equal(expected, null_equal=True)


def test_transpose() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    expected = pl.DataFrame(
        {
            "column": ["a", "b"],
            "column_0": [1, 1],
            "column_1": [2, 2],
            "column_2": [3, 3],
        }
    )
    out = df.transpose(include_header=True)
    assert expected.frame_equal(out)

    out = df.transpose(include_header=False, column_names=["a", "b", "c"])
    expected = pl.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert expected.frame_equal(out)

    out = df.transpose(
        include_header=True, header_name="foo", column_names=["a", "b", "c"]
    )
    expected = pl.DataFrame(
        {
            "foo": ["a", "b"],
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert expected.frame_equal(out)

    def name_generator() -> tp.Iterator[str]:
        base_name = "my_column_"
        count = 0
        while True:
            yield f"{base_name}{count}"
            count += 1

    out = df.transpose(include_header=False, column_names=name_generator())
    expected = pl.DataFrame(
        {
            "my_column_0": [1, 1],
            "my_column_1": [2, 2],
            "my_column_2": [3, 3],
        }
    )
    assert expected.frame_equal(out)


def test_extension() -> None:
    class Foo:
        def __init__(self, value):  # type: ignore
            self.value = value

        def __repr__(self):  # type: ignore
            return f"foo({self.value})"

    foos = [Foo(1), Foo(2), Foo(3)]
    # I believe foos, stack, and sys.getrefcount have a ref
    base_count = 3
    assert sys.getrefcount(foos[0]) == base_count

    df = pl.DataFrame({"groups": [1, 1, 2], "a": foos})
    assert sys.getrefcount(foos[0]) == base_count + 1
    del df
    assert sys.getrefcount(foos[0]) == base_count

    df = pl.DataFrame({"groups": [1, 1, 2], "a": foos})
    assert sys.getrefcount(foos[0]) == base_count + 1

    out = df.groupby("groups", maintain_order=True).agg(pl.col("a").list().alias("a"))
    assert sys.getrefcount(foos[0]) == base_count + 2
    s = out["a"].explode()
    assert sys.getrefcount(foos[0]) == base_count + 3
    del s
    assert sys.getrefcount(foos[0]) == base_count + 2

    assert out["a"].explode().to_list() == foos
    assert sys.getrefcount(foos[0]) == base_count + 2
    del out
    assert sys.getrefcount(foos[0]) == base_count + 1
    del df
    assert sys.getrefcount(foos[0]) == base_count


def test_schema() -> None:
    df = pl.DataFrame(
        {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
    )
    expected = {"foo": pl.Int64, "bar": pl.Float64, "ham": pl.Utf8}
    assert df.schema == expected


def test_df_schema_unique() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(Exception):
        df.columns = ["a", "a"]

    with pytest.raises(Exception):
        df.rename({"b": "a"})


def test_empty_projection() -> None:
    assert pl.DataFrame({"a": [1, 2], "b": [3, 4]}).select([]).shape == (0, 0)
