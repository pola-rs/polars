from __future__ import annotations

from datetime import datetime
from typing import Any, cast, no_type_check
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl


def test_df_from_numpy() -> None:
    df = pl.DataFrame(
        {
            "int8": np.array([1, 3, 2], dtype=np.int8),
            "int16": np.array([1, 3, 2], dtype=np.int16),
            "int32": np.array([1, 3, 2], dtype=np.int32),
            "int64": np.array([1, 3, 2], dtype=np.int64),
            "uint8": np.array([1, 3, 2], dtype=np.uint8),
            "uint16": np.array([1, 3, 2], dtype=np.uint16),
            "uint32": np.array([1, 3, 2], dtype=np.uint32),
            "uint64": np.array([1, 3, 2], dtype=np.uint64),
            "float16": np.array([21.7, 21.8, 21], dtype=np.float16),
            "float32": np.array([21.7, 21.8, 21], dtype=np.float32),
            "float64": np.array([21.7, 21.8, 21], dtype=np.float64),
            "str": np.array(["string1", "string2", "string3"], dtype=np.str_),
            "bytes": np.array(
                ["byte_string1", "byte_string2", "byte_string3"], dtype=np.bytes_
            ),
        }
    )
    out = [
        pl.datatypes.Int8,
        pl.datatypes.Int16,
        pl.datatypes.Int32,
        pl.datatypes.Int64,
        pl.datatypes.UInt8,
        pl.datatypes.UInt16,
        pl.datatypes.UInt32,
        pl.datatypes.UInt64,
        # np.float16 gets converted to float32 as Rust does not support float16.
        pl.datatypes.Float32,
        pl.datatypes.Float32,
        pl.datatypes.Float64,
        pl.datatypes.Utf8,
        pl.datatypes.Object,
    ]
    assert out == df.dtypes


def test_to_numpy() -> None:
    def test_series_to_numpy(
        name: str,
        values: list[object],
        pl_dtype: type[pl.DataType],
        np_dtype: (
            type[np.signedinteger[Any]]
            | type[np.unsignedinteger[Any]]
            | type[np.floating[Any]]
            | type[np.object_]
        ),
    ) -> None:
        pl_series_to_numpy_array = np.array(pl.Series(name, values, pl_dtype))
        numpy_array = np.array(values, dtype=np_dtype)
        assert pl_series_to_numpy_array.dtype == numpy_array.dtype
        assert np.all(pl_series_to_numpy_array == numpy_array) == np.bool_(True)

    test_series_to_numpy("int8", [1, 3, 2], pl.Int8, np.int8)
    test_series_to_numpy("int16", [1, 3, 2], pl.Int16, np.int16)
    test_series_to_numpy("int32", [1, 3, 2], pl.Int32, np.int32)
    test_series_to_numpy("int64", [1, 3, 2], pl.Int64, np.int64)

    test_series_to_numpy("uint8", [1, 3, 2], pl.UInt8, np.uint8)
    test_series_to_numpy("uint16", [1, 3, 2], pl.UInt16, np.uint16)
    test_series_to_numpy("uint32", [1, 3, 2], pl.UInt32, np.uint32)
    test_series_to_numpy("uint64", [1, 3, 2], pl.UInt64, np.uint64)

    test_series_to_numpy("float32", [21.7, 21.8, 21], pl.Float32, np.float32)
    test_series_to_numpy("float64", [21.7, 21.8, 21], pl.Float64, np.float64)

    test_series_to_numpy("str", ["string1", "string2", "string3"], pl.Utf8, np.object_)


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

    df = pd.Series([2, np.nan, None], name="pd")  # type: ignore[assignment]
    out_true = pl.from_pandas(df)
    out_false = pl.from_pandas(df, nan_to_none=False)
    df.loc[2] = pd.NA
    assert [val is None for val in out_true]
    assert [np.isnan(val) for val in out_false[1:]]
    with pytest.raises(ArrowInvalid, match="Could not convert"):
        pl.from_pandas(df, nan_to_none=False)


def test_from_pandas_datetime() -> None:
    ts = datetime(2021, 1, 1, 20, 20, 20, 20)
    pl_s = pd.Series([ts, ts])
    tmp = pl.from_pandas(pl_s.to_frame("a"))
    s = tmp["a"]
    assert s.dt.hour()[0] == 20
    assert s.dt.minute()[0] == 20
    assert s.dt.second()[0] == 20

    date_times = pd.date_range("2021-06-24 00:00:00", "2021-06-24 09:00:00", freq="1H")
    s = pl.from_pandas(date_times)
    assert s[0] == datetime(2021, 6, 24, 0, 0)
    assert s[-1] == datetime(2021, 6, 24, 9, 0)

    df = pd.DataFrame({"datetime": ["2021-01-01", "2021-01-02"], "foo": [1, 2]})
    df["datetime"] = pd.to_datetime(df["datetime"])
    pl.from_pandas(df)


def test_from_pandas_duplicated_columns() -> None:
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["a", "b", "c", "b"])
    with pytest.raises(ValueError, match="Duplicate column names found: "):
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
    s = cast(pl.Series, pl.from_arrow(ca))
    assert s.dtype == pl.List


def test_from_pandas_null() -> None:
    # null column is an object dtype, so pl.Utf8 is most close
    df = pd.DataFrame([{"a": None}, {"a": None}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.Utf8]
    assert out["a"][0] is None

    df = pd.DataFrame([{"a": None, "b": 1}, {"a": None, "b": 2}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.Utf8, pl.Int64]


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
    df = pl.from_dict(data)
    assert df.shape == (2, 2)


def test_from_dict_struct() -> None:
    data: dict[str, dict[str, list[int]] | list[int]] = {
        "a": {"b": [1, 3], "c": [2, 4]},
        "d": [5, 6],
    }
    df = pl.from_dict(data)
    assert df.shape == (2, 2)
    assert df["a"][0] == {"b": 1, "c": 2}
    assert df["a"][1] == {"b": 3, "c": 4}


def test_from_dicts() -> None:
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    df = pl.from_dicts(data)
    assert df.shape == (3, 2)


@pytest.mark.skip("not implemented yet")
def test_from_dicts_struct() -> None:
    data = [{"a": {"b": 1, "c": 2}, "d": 5}, {"a": {"b": 3, "c": 4}, "d": 6}]
    df = pl.from_dicts(data)
    assert df.shape == (2, 2)
    assert df["a"][0] == (1, 2)
    assert df["a"][1] == (3, 4)


def test_from_records() -> None:
    data = [[1, 2, 3], [4, 5, 6]]
    df = pl.from_records(data, columns=["a", "b"])
    assert df.shape == (3, 2)


def test_from_numpy() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    df = pl.from_numpy(data, columns=["a", "b"], orient="col")
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

    # if not a pandas dataframe, raise a ValueError
    with pytest.raises(ValueError):
        _ = pl.from_pandas([1, 2])  # type: ignore[call-overload]


def test_from_pandas_series() -> None:
    pd_series = pd.Series([1, 2, 3], name="pd")
    df = pl.from_pandas(pd_series)
    assert df.shape == (3,)


def test_from_optional_not_available() -> None:
    with patch("polars.convert._NUMPY_AVAILABLE", False):
        with pytest.raises(ImportError):
            pl.from_numpy(np.array([[1, 2], [3, 4]]), columns=["a", "b"])
    with patch("polars.convert._PYARROW_AVAILABLE", False):
        with pytest.raises(ImportError):
            pl.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
        with pytest.raises(ImportError):
            pl.from_pandas(pd.Series([1, 2, 3]))
    with patch("polars.convert._PANDAS_AVAILABLE", False):
        with pytest.raises(ImportError):
            pl.from_pandas(pd.Series([1, 2, 3]))


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
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Float64, pl.Float64]

    # 2705
    df1 = pd.DataFrame(columns=["b"], dtype=float)
    tbl = pa.Table.from_pandas(df1)
    out = pl.from_arrow(tbl)
    assert out.columns == ["b", "__index_level_0__"]
    assert out.dtypes == [pl.Float64, pl.Int8]
    tbl = pa.Table.from_pandas(df1, preserve_index=False)
    out = pl.from_arrow(tbl)
    assert out.columns == ["b"]
    assert out.dtypes == [pl.Float64]

    # 4568
    tbl = pa.table({"l": []}, schema=pa.schema([("l", pa.large_list(pa.uint8()))]))

    df = pl.from_arrow(tbl)
    assert df.schema["l"] == pl.List(pl.UInt8)


def test_from_null_column() -> None:
    assert pl.from_pandas(pd.DataFrame(data=[pd.NA, pd.NA])).shape == (2, 1)


def test_to_pandas_series() -> None:
    assert (pl.Series("a", [1, 2, 3]).to_pandas() == pd.Series([1, 2, 3])).all()


def test_respect_dtype_with_series_from_numpy() -> None:
    assert pl.Series("foo", np.array([1, 2, 3]), dtype=pl.UInt32).dtype == pl.UInt32


def test_from_pandas_ns_resolution() -> None:
    df = pd.DataFrame(
        [pd.Timestamp(year=2021, month=1, day=1, hour=1, second=1, nanosecond=1)],
        columns=["date"],
    )
    assert cast(datetime, pl.from_pandas(df)[0, 0]) == datetime(2021, 1, 1, 1, 0, 1)


@no_type_check
def test_pandas_string_none_conversion_3298() -> None:
    data = {"col_1": ["a", "b", "c", "d"]}
    data["col_1"][0] = None
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(df_pd)
    assert df_pl.to_series().to_list() == [None, "b", "c", "d"]


def test_cat_int_types_3500() -> None:
    with pl.StringCache():
        # Create an enum / categorical / dictionary typed pyarrow array
        # Most simply done by creating a pandas categorical series first
        categorical_s = pd.Series(["a", "a", "b"], dtype="category")
        pyarrow_array = pa.Array.from_pandas(categorical_s)

        # The in-memory representation of each category can either be a signed or
        # unsigned 8-bit integer. Pandas uses Int8...
        int_dict_type = pa.dictionary(index_type=pa.int8(), value_type=pa.utf8())
        # ... while DuckDB uses UInt8
        uint_dict_type = pa.dictionary(index_type=pa.uint8(), value_type=pa.utf8())

        for t in [int_dict_type, uint_dict_type]:
            s = cast(pl.Series, pl.from_arrow(pyarrow_array.cast(t)))
            assert s.series_equal(pl.Series(["a", "a", "b"]).cast(pl.Categorical))


def test_from_pyarrow_chunked_array() -> None:
    column = pa.chunked_array([[1], [2]])
    series = pl.Series("column", column)
    assert series.to_list() == [1, 2]


def test_numpy_preserve_uint64_4112() -> None:
    assert pl.DataFrame({"a": [1, 2, 3]}).with_column(
        pl.col("a").hash()
    ).to_numpy().dtype == np.dtype("uint64")


def test_view_ub() -> None:
    # this would be UB if the series was dropped and not passed to the view
    assert np.sum(pl.Series([3, 1, 5]).sort().view()) == 9
