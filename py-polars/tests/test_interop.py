import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl


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
    assert all(val is None for val in out_true["nulls"])
    assert all(np.isnan(val) for val in out_false["nulls"][1:])
    with pytest.raises(ArrowInvalid, match="Could not convert"):
        pl.from_pandas(df, nan_to_none=False)

    df = pd.Series([2, np.nan, None], name="pd")  # type: ignore
    out_true = pl.from_pandas(df)
    out_false = pl.from_pandas(df, nan_to_none=False)
    df.loc[2] = pd.NA
    assert [val is None for val in out_true]
    assert [np.isnan(val) for val in out_false[1:]]
    with pytest.raises(ArrowInvalid, match="Could not convert"):
        pl.from_pandas(df, nan_to_none=False)


def test_from_pandas_datetime() -> None:
    ts = datetime.datetime(2021, 1, 1, 20, 20, 20, 20)
    pl_s = pd.Series([ts, ts])
    tmp = pl.from_pandas(pl_s.to_frame("a"))
    s = tmp["a"]
    assert s.dt.hour()[0] == 20
    assert s.dt.minute()[0] == 20
    assert s.dt.second()[0] == 20

    date_times = pd.date_range(
        "2021-06-24 00:00:00", "2021-06-24 10:00:00", freq="1H", closed="left"
    )
    s = pl.from_pandas(date_times)
    assert s[0] == datetime.datetime(2021, 6, 24, 0, 0)
    assert s[-1] == datetime.datetime(2021, 6, 24, 9, 0)

    df = pd.DataFrame({"datetime": ["2021-01-01", "2021-01-02"], "foo": [1, 2]})
    df["datetime"] = pd.to_datetime(df["datetime"])
    pl.from_pandas(df)


def test_arrow_list_roundtrip() -> None:
    # https://github.com/pola-rs/polars/issues/1064
    tbl = pa.table({"a": [1], "b": [[1, 2]]})
    assert pl.from_arrow(tbl).to_arrow().shape == tbl.shape


def test_arrow_dict_to_polars() -> None:
    pa_dict = pa.DictionaryArray.from_arrays(
        indices=np.array([0, 1, 2, 3, 1, 0, 2, 3, 3, 2]),
        dictionary=np.array(["AAA", "BBB", "CCC", "DDD"]),
    ).cast(pa.large_utf8())

    s = pl.Series(
        name="pa_dict",
        values=["AAA", "BBB", "CCC", "DDD", "BBB", "AAA", "CCC", "DDD", "DDD", "CCC"],
    )

    assert s.series_equal(pl.Series("pa_dict", pa_dict))


def test_arrow_list_chunked_array() -> None:
    a = pa.array([[1, 2], [3, 4]])
    ca = pa.chunked_array([a, a, a])
    s = pl.from_arrow(ca)
    assert s.dtype == pl.List


def test_from_pandas_null() -> None:
    df = pd.DataFrame([{"a": None}, {"a": None}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.Float64]
    assert out["a"][0] is None

    df = pd.DataFrame([{"a": None, "b": 1}, {"a": None, "b": 2}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.Float64, pl.Int64]


def test_from_pandas_nested_list() -> None:
    # this panicked in https://github.com/pola-rs/polars/issues/1615
    pddf = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [["x", "y"], ["x", "y", "z"], ["x"], ["x", "y"]]}
    )
    pldf = pl.from_pandas(pddf)
    assert pldf.shape == (4, 2)


def test_from_pandas_categorical_none() -> None:
    s = pd.Series(["a", "b", "c", pd.NA], dtype="category")
    out = pl.from_pandas(s)
    assert out.dtype == pl.Categorical
    assert out.to_list() == ["a", "b", "c", None]


def test_from_dict() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.from_dict(data)  # type: ignore
    assert df.shape == (2, 2)


def test_from_dicts() -> None:
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    df = pl.from_dicts(data)
    assert df.shape == (3, 2)


def test_from_records() -> None:
    data = [[1, 2, 3], [4, 5, 6]]
    df = pl.from_records(data, columns=["a", "b"])
    assert df.shape == (3, 2)


def test_from_arrow() -> None:
    data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = pl.from_arrow(data)
    assert df.shape == (3, 2)

    # if not a PyArrow type, raise a ValueError
    with pytest.raises(ValueError):
        _ = pl.from_arrow([1, 2])


def test_from_pandas_dataframe() -> None:
    pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    df = pl.from_pandas(pd_df)
    assert df.shape == (2, 3)

    # if not a Pandas dataframe, raise a ValueError
    with pytest.raises(ValueError):
        _ = pl.from_pandas([1, 2])  # type: ignore


def test_from_pandas_series() -> None:
    pd_series = pd.Series([1, 2, 3], name="pd")
    df = pl.from_pandas(pd_series)
    assert df.shape == (3,)


def test_upcast_pyarrow_dicts() -> None:
    # 1752
    tbls = []
    for i in range(128):
        tbls.append(
            pa.table(
                {
                    "col_name": pa.array(
                        ["value_" + str(i)], pa.dictionary(pa.int8(), pa.string())
                    ),
                }
            )
        )

    tbl = pa.concat_tables(tbls, promote=True)
    out = pl.from_arrow(tbl)
    assert out.shape == (128, 1)


def test_no_rechunk() -> None:
    table = pa.Table.from_pydict({"x": pa.chunked_array([list("ab"), list("cd")])})
    # table
    assert pl.from_arrow(table, rechunk=False).n_chunks() == 2
    # chunked array
    assert pl.from_arrow(table["x"], rechunk=False).n_chunks() == 2


def test_cat_to_pandas() -> None:
    df = pl.DataFrame({"a": ["best", "test"]})
    df = df.with_columns(pl.all().cast(pl.Categorical))
    out = df.to_pandas()
    assert "category" in str(out["a"].dtype)


def test_numpy_to_lit() -> None:
    out = pl.select(pl.lit(np.array([1, 2, 3]))).to_series().to_list()
    assert out == [1, 2, 3]
    out = pl.select(pl.lit(np.float32(0))).to_series().to_list()
    assert out == [0.0]


def test_from_empty_pandas() -> None:
    pandas_df = pd.DataFrame(
        {
            "A": [],
            "fruits": [],
        }
    )

    polars_df = pl.from_pandas(pandas_df)
    assert polars_df.columns == ["A", "fruits"]
    assert polars_df.dtypes == [pl.Float64, pl.Float64]


def test_from_empty_pandas_strings() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    df["a"] = df["a"].astype(str)
    df["b"] = df["b"].astype(float)
    df_pl = pl.from_pandas(df)
    assert df_pl.dtypes == [pl.Utf8, pl.Float64]


def test_from_empty_arrow() -> None:
    df = pl.from_arrow(pa.table(pd.DataFrame({"a": [], "b": []})))
    assert df.columns == ["a", "b"]  # type: ignore
    assert df.dtypes == [pl.Float64, pl.Float64]  # type: ignore

    # 2705
    df1 = pd.DataFrame(columns=["b"], dtype=float)
    tbl = pa.Table.from_pandas(df1)
    out = pl.from_arrow(tbl)
    assert out.columns == ["b", "__index_level_0__"]  # type: ignore
    assert out.dtypes == [pl.Float64, pl.Utf8]  # type: ignore
    tbl = pa.Table.from_pandas(df1, preserve_index=False)
    out = pl.from_arrow(tbl)
    assert out.columns == ["b"]  # type: ignore
    assert out.dtypes == [pl.Float64]  # type: ignore


def test_from_null_column() -> None:
    assert pl.from_pandas(pd.DataFrame(data=[pd.NA, pd.NA])).shape == (2, 1)


def test_to_pandas_series() -> None:
    assert (pl.Series("a", [1, 2, 3]).to_pandas() == pd.Series([1, 2, 3])).all()
