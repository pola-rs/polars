from polars import DataFrame, Series
from polars.datatypes import *
from polars.lazy import *
from polars import functions
import pytest
from io import BytesIO
import numpy as np
from builtins import range
import pyarrow as pa
import polars as pl
import pandas as pd

from utils import get_complete_df


def test_version():
    pl.__version__


def test_init():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.shape == (3, 2)

    # length mismatch
    with pytest.raises(RuntimeError):
        pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0, 4.0]})

    df = pl.DataFrame(np.random.randn(3, 5))
    assert df.shape == (3, 5)
    assert df.columns == ["0", "1", "2", "3", "4"]


def test_selection():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    assert df["a"].dtype == Int64
    assert df["b"].dtype == Float64
    assert df["c"].dtype == Utf8

    assert df[["a", "b"]].columns == ["a", "b"]
    assert df[[True, False, True]].height == 2

    assert df[[True, False, True], "b"].shape == (2, 1)
    assert df[[True, False, False], ["a", "b"]].shape == (1, 2)

    assert df[[0, 1], "b"].shape == (2, 1)
    assert df[[2], ["a", "b"]].shape == (1, 2)
    assert df.select_at_idx(0).name == "a"
    assert (df.a == df["a"]).sum() == 3
    assert (df.c == df["a"]).sum() == 0
    assert df[:, "a":"b"].shape == (3, 2)


def test_from_arrow():
    tbl = pa.table(
        {
            "a": pa.array([1, 2], pa.timestamp("s")),
            "b": pa.array([1, 2], pa.timestamp("ms")),
            "c": pa.array([1, 2], pa.timestamp("us")),
            "d": pa.array([1, 2], pa.timestamp("ns")),
            "decimal1": pa.array([1, 2], pa.decimal128(2, 2)),
        }
    )
    assert pl.from_arrow_table(tbl).shape == (2, 5)


def test_downsample():
    s = Series(
        "datetime",
        [
            946684800000,
            946684860000,
            946684920000,
            946684980000,
            946685040000,
            946685100000,
            946685160000,
            946685220000,
            946685280000,
            946685340000,
            946685400000,
            946685460000,
            946685520000,
            946685580000,
            946685640000,
            946685700000,
            946685760000,
            946685820000,
            946685880000,
            946685940000,
        ],
    ).cast(Date64)
    s2 = s.clone()
    df = DataFrame({"a": s, "b": s2})
    out = df.downsample("a", rule="minute", n=5).first()
    assert out.shape == (4, 2)

    # OLHC
    out = df.downsample("a", rule="minute", n=5).agg(
        {"b": ["first", "min", "max", "last"]}
    )
    assert out.shape == (4, 5)

    # test to_pandas as well.
    out = df.to_pandas()
    assert out["a"].dtype == "datetime64[ns]"


def test_sort():
    df = DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    df.sort("a", in_place=True)
    assert df.frame_equal(DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]}))


def test_replace():
    df = DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    s = Series("c", [True, False, True])
    df.replace("a", s)
    assert df.frame_equal(DataFrame({"c": [True, False, True], "b": [1, 2, 3]}))


def test_slice():
    df = DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df = df.slice(1, 2)
    assert df.frame_equal(DataFrame({"a": [1, 3], "b": ["b", "c"]}))


def test_null_count():
    df = DataFrame({"a": [2, 1, 3], "b": ["a", "b", None]})
    assert df.null_count().shape == (1, 2)


def test_head_tail():
    df = DataFrame({"a": range(10), "b": range(10)})
    assert df.head(5).height == 5
    assert df.tail(5).height == 5

    assert not df.head(5).frame_equal(df.tail(5))
    # check if it doesn't fail when out of bounds
    assert df.head(100).height == 10
    assert df.tail(100).height == 10


def test_groupby():
    df = DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    # use __getitem__ to map to select
    assert (
        df.groupby("a")["b"]
        .sum()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [4, 11, 6]}))
    )

    assert (
        df.groupby("a")
        .select("b")
        .sum()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [4, 11, 6]}))
    )
    assert (
        df.groupby("a")
        .select("c")
        .sum()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [10, 10, 1]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .min()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [1, 2, 6]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .max()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [3, 5, 6]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .mean()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [2.0, (2 + 4 + 5) / 3, 6.0]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .last()
        .sort(by_column="a")
        .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [3, 5, 6]}))
    )
    # check if it runs
    (df.groupby("a").select("b").n_unique())

    (df.groupby("a").select("b").quantile(0.3))
    (df.groupby("a").select("b").agg_list())

    gb_df = df.groupby("a").agg({"b": ["sum", "min"], "c": "count"})
    assert "b_sum" in gb_df.columns
    assert "b_min" in gb_df.columns

    #
    # # TODO: is false because count is u32
    # df.groupby(by="a", select="b", agg="count").frame_equal(
    #     DataFrame({"a": ["a", "b", "c"], "": [2, 3, 1]})
    # )
    assert df.groupby("a").apply(lambda df: df[["c"]].sum()).sort("c")["c"][0] == 1

    assert df.groupby("a").groups().sort("a")["a"].series_equal(Series(["a", "b", "c"]))

    for subdf in df.groupby("a"):
        if subdf["a"][0] == "b":
            assert subdf.shape == (3, 3)

    assert df.groupby("a").get_group("c").shape == (1, 3)
    assert df.groupby("a").get_group("b").shape == (3, 3)
    assert df.groupby("a").get_group("a").shape == (2, 3)

    # Use lazy API in eager groupby
    assert df.groupby("a").agg([pl.sum("b")]).shape == (3, 2)


def test_join():
    df_left = DataFrame(
        {
            "a": ["a", "b", "a", "z"],
            "b": [1, 2, 3, 4],
            "c": [6, 5, 4, 3],
        }
    )
    df_right = DataFrame(
        {
            "a": ["b", "c", "b", "a"],
            "k": [0, 3, 9, 6],
            "c": [1, 0, 2, 1],
        }
    )

    joined = df_left.join(df_right, left_on="a", right_on="a").sort("a")
    assert joined["b"].series_equal(Series("", [1, 3, 2, 2]))
    joined = df_left.join(df_right, left_on="a", right_on="a", how="left").sort("a")
    assert joined["c_right"].is_null().sum() == 1
    assert joined["b"].series_equal(Series("", [1, 3, 2, 2, 4]))
    joined = df_left.join(df_right, left_on="a", right_on="a", how="outer").sort("a")
    assert joined["c_right"].null_count() == 1
    assert joined["c"].null_count() == 2
    assert joined["b"].null_count() == 2

    df_a = DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})
    df_b = DataFrame(
        {"foo": [1, 1, 1], "bar": ["a", "c", "c"], "ham": ["let", "var", "const"]}
    )

    # just check if join on multiple columns runs
    df_a.join(df_b, left_on=["a", "b"], right_on=["foo", "bar"])

    eager_join = df_a.join(df_b, left_on="a", right_on="foo")

    lazy_join = df_a.lazy().join(df_b.lazy(), left_on="a", right_on="foo").collect()
    assert lazy_join.shape == eager_join.shape


def test_hstack():
    df = DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df.hstack([Series("stacked", [-1, -1, -1])], in_place=True)
    assert df.shape == (3, 3)
    assert df.columns == ["a", "b", "stacked"]


def test_drop():
    df = DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    df = df.drop("a")
    assert df.shape == (3, 2)
    df = DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    s = df.drop_in_place("a")
    assert s.name == "a"


def test_file_buffer():
    f = BytesIO()
    f.write(b"1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    df = DataFrame.read_csv(f, has_headers=False)
    assert df.shape == (2, 6)
    f.seek(0)

    # check if not fails on TryClone and Length impl in file.rs
    with pytest.raises(RuntimeError) as e:
        df.read_parquet(f)
    assert "Invalid Parquet file" in str(e.value)


def test_set():
    np.random.seed(1)
    df = DataFrame({"foo": np.random.rand(10), "bar": np.arange(10), "ham": ["h"] * 10})
    df["new"] = np.random.rand(10)
    df[df["new"] > 0.5, "new"] = 1


def test_melt():
    df = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    melted = df.melt(id_vars="A", value_vars=["B", "C"])
    assert melted["value"] == [1, 3, 4, 2, 4, 6]


def test_shift():
    df = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    a = df.shift(1)
    b = DataFrame({"A": [None, "a", "b"], "B": [None, 1, 3]}, nullable=True)
    assert a.frame_equal(b, null_equal=True)


def test_to_dummies():
    df = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    dummies = df.to_dummies()
    assert dummies["A_a"].to_list() == [1, 0, 0]
    assert dummies["A_b"].to_list() == [0, 1, 0]
    assert dummies["A_c"].to_list() == [0, 0, 1]


def test_from_pandas():
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


def test_custom_groupby():
    df = DataFrame({"A": ["a", "a", "c", "c"], "B": [1, 3, 5, 2]})
    assert df.groupby("A").select("B").apply(lambda x: x.sum()).shape == (2, 2)
    assert df.groupby("A").select("B").apply(
        lambda x: Series("", np.array(x))
    ).shape == (
        2,
        2,
    )

    df = DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})

    out = (
        df.lazy()
        .groupby("b")
        .agg([col("a").apply(lambda x: x.sum(), dtype_out=int)])
        .collect()
    )
    assert out.shape == (3, 2)


def test_multiple_columns_drop():
    df = DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    out = df.drop(["a", "b"])
    assert out.columns == ["c"]


def test_concat():
    df = DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    assert functions.concat([df, df]).shape == (6, 3)


def test_to_pandas():
    df = get_complete_df()
    df.to_arrow()
    df.to_pandas()
    # test shifted df
    df.shift(2).to_pandas()
    df = DataFrame({"col": Series([True, False, True])})
    df.shift(2).to_pandas()


def test_from_arrow_table():
    data = {"a": [1, 2], "b": [1, 2]}
    tbl = pa.table(data)

    df = pl.from_arrow_table(tbl)
    df.frame_equal(pl.DataFrame(data))


def test_df_stats():
    df = get_complete_df()
    df.var()
    df.std()
    df.min()
    df.max()
    df.sum()
    df.mean()
    df.median()
    df.quantile(0.4)


def test_from_pandas_datetime():
    df = pd.DataFrame({"datetime": ["2021-01-01", "2021-01-02"], "foo": [1, 2]})
    df["datetime"] = pd.to_datetime(df["datetime"])
    pl.from_pandas(df)


def test_df_fold():
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    assert df.fold(lambda s1, s2: s1 + s2).series_equal(Series("a", [4.0, 5.0, 9.0]))
    assert df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2)).series_equal(
        Series("a", [1.0, 1.0, 3.0])
    )

    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.fold(lambda s1, s2: s1 + s2)
    out.series_equal(Series("", ["foo11", "bar22", "233"]))

    df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    # just check dispatch. values are tested on rust side.
    assert df.sum(axis=1).shape == (3, 1)
    assert df.mean(axis=1).shape == (3, 1)
    assert df.min(axis=1).shape == (3, 1)
    assert df.max(axis=1).shape == (3, 1)


def test_row_tuple():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    assert df.row(0) == ("foo", 1, 1.0)
    assert df.row(1) == ("bar", 2, 2.0)
    assert df.row(-1) == ("2", 3, 3.0)


def test_read_csv_categorical():
    f = BytesIO()
    f.write(b"col1,col2,col3,col4,col5,col6\n'foo',2,3,4,5,6\n'bar',8,9,10,11,12")
    f.seek(0)
    df = pl.DataFrame.read_csv(f, has_headers=True, dtype={"col1": pl.Categorical})
    assert df["col1"].dtype == pl.Categorical


def test_df_apply():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.apply(lambda x: len(x), None)
    assert out.sum() == 9


def test_column_names():
    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5], pa.decimal128(38, 2)),
            "b": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    )
    df = pl.from_arrow(tbl)
    assert df.columns == ["a", "b"]


def test_lazy_functions():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df[[pl.count("a")]]
    assert out[0] == 3
    assert pl.count(df["a"]) == 3
    out = df[
        [
            pl.var("b"),
            pl.std("b"),
            pl.max("b"),
            pl.min("b"),
            pl.sum("b"),
            pl.mean("b"),
            pl.median("b"),
            pl.n_unique("b"),
            pl.first("b"),
            pl.last("b"),
        ]
    ]
    expected = 1.0
    assert np.isclose(out[0], expected)
    assert np.isclose(pl.var(df["b"]), expected)
    expected = 1.0
    assert np.isclose(out[1], expected)
    assert np.isclose(pl.std(df["b"]), expected)
    expected = 3
    assert np.isclose(out[2], expected)
    assert np.isclose(pl.max(df["b"]), expected)
    expected = 1
    assert np.isclose(out[3], expected)
    assert np.isclose(pl.min(df["b"]), expected)
    expected = 6
    assert np.isclose(out[4], expected)
    assert np.isclose(pl.sum(df["b"]), expected)
    expected = 2
    assert np.isclose(out[5], expected)
    assert np.isclose(pl.mean(df["b"]), expected)
    expected = 2
    assert np.isclose(out[6], expected)
    assert np.isclose(pl.median(df["b"]), expected)
    expected = 3
    assert np.isclose(out[7], expected)
    assert np.isclose(pl.n_unique(df["b"]), expected)
    expected = 1
    assert np.isclose(out[8], expected)
    assert np.isclose(pl.first(df["b"]), expected)
    expected = 3
    assert np.isclose(out[9], expected)
    assert np.isclose(pl.last(df["b"]), expected)
    expected = 3
    assert np.isclose(out[9], expected)
    assert np.isclose(pl.last(df["b"]), expected)
