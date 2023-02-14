from __future__ import annotations

import typing
from datetime import datetime
from typing import Any, cast, no_type_check

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.datatypes import dtype_to_py_type
from polars.testing import assert_series_equal


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
            "intc": np.array([1, 3, 2], dtype=np.intc),
            "uintc": np.array([1, 3, 2], dtype=np.uintc),
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
        pl.datatypes.Int32,
        pl.datatypes.UInt32,
        pl.datatypes.Utf8,
        pl.datatypes.Binary,
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
    # without pyarrow
    arr = pl.Series(["a", "b", None]).to_numpy(use_pyarrow=False)
    assert arr.dtype == np.dtype("O")
    assert list(arr) == ["a", "b", None]


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
    assert out.schema == {
        "bools": pl.Boolean,
        "bools_nulls": pl.Boolean,
        "int": pl.Int64,
        "int_nulls": pl.Float64,
        "floats": pl.Float64,
        "floats_nulls": pl.Float64,
        "strings": pl.Utf8,
        "strings_nulls": pl.Utf8,
        "strings-cat": pl.Categorical,
    }
    assert out.rows() == [
        (False, None, 1, 1.0, 1.0, 1.0, "foo", "foo", "foo"),
        (True, True, 2, None, 2.0, None, "bar", None, "bar"),
        (False, False, 3, 3.0, 3.0, 3.0, "ham", "ham", "ham"),
    ]

    # partial dtype overrides from pandas
    overrides = {"int": pl.Int8, "int_nulls": pl.Int32, "floats": pl.Float32}
    out = pl.from_pandas(df, schema_overrides=overrides)
    for col, dtype in overrides.items():
        assert out.schema[col] == dtype


def test_from_pandas_nan_to_null() -> None:
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
    out_false = pl.from_pandas(df, nan_to_null=False)
    assert all(val is None for val in out_true["nulls"])
    assert all(np.isnan(val) for val in out_false["nulls"][1:])

    df = pd.Series([2, np.nan, None], name="pd")  # type: ignore[assignment]
    out_true = pl.from_pandas(df)
    out_false = pl.from_pandas(df, nan_to_null=False)
    assert [val is None for val in out_true]
    assert [np.isnan(val) for val in out_false[1:]]


def test_from_pandas_datetime() -> None:
    ts = datetime(2021, 1, 1, 20, 20, 20, 20)
    pd_s = pd.Series([ts, ts])
    tmp = pl.from_pandas(pd_s.to_frame("a"))
    s = tmp["a"]
    assert s.dt.hour()[0] == 20
    assert s.dt.minute()[0] == 20
    assert s.dt.second()[0] == 20

    date_times = pd.date_range("2021-06-24 00:00:00", "2021-06-24 09:00:00", freq="1H")
    s = pl.from_pandas(date_times)
    assert s[0] == datetime(2021, 6, 24, 0, 0)
    assert s[-1] == datetime(2021, 6, 24, 9, 0)


def test_from_pandas_include_indexes() -> None:
    data = {
        "dtm": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        "val": [100, 200, 300],
        "misc": ["x", "y", "z"],
    }
    pd_df = pd.DataFrame(data)

    df = pl.from_pandas(pd_df.set_index(["dtm"]))
    assert df.to_dict(False) == {"val": [100, 200, 300], "misc": ["x", "y", "z"]}

    df = pl.from_pandas(pd_df.set_index(["dtm", "val"]))
    assert df.to_dict(False) == {"misc": ["x", "y", "z"]}

    df = pl.from_pandas(pd_df.set_index(["dtm"]), include_index=True)
    assert df.to_dict(False) == data

    df = pl.from_pandas(pd_df.set_index(["dtm", "val"]), include_index=True)
    assert df.to_dict(False) == data


def test_from_pandas_duplicated_columns() -> None:
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["a", "b", "c", "b"])
    with pytest.raises(ValueError, match="Duplicate column names found: "):
        pl.from_pandas(df)


def test_arrow_list_roundtrip() -> None:
    # https://github.com/pola-rs/polars/issues/1064
    tbl = pa.table({"a": [1], "b": [[1, 2]]})
    arw = pl.from_arrow(tbl).to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_dict_to_polars() -> None:
    pa_dict = pa.DictionaryArray.from_arrays(
        indices=np.array([0, 1, 2, 3, 1, 0, 2, 3, 3, 2]),
        dictionary=np.array(["AAA", "BBB", "CCC", "DDD"]),
    ).cast(pa.large_utf8())

    s = pl.Series(
        name="pa_dict",
        values=["AAA", "BBB", "CCC", "DDD", "BBB", "AAA", "CCC", "DDD", "DDD", "CCC"],
    )

    assert_series_equal(s, pl.Series("pa_dict", pa_dict))


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
    assert pldf.rows() == [
        (1, ["x", "y"]),
        (2, ["x", "y", "z"]),
        (3, ["x"]),
        (4, ["x", "y"]),
    ]


def test_from_pandas_categorical_none() -> None:
    s = pd.Series(["a", "b", "c", pd.NA], dtype="category")
    out = pl.from_pandas(s)
    assert out.dtype == pl.Categorical
    assert out.to_list() == ["a", "b", "c", None]


def test_from_dict() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.from_dict(data)
    assert df.shape == (2, 2)
    for s1, s2 in zip(list(df), [pl.Series("a", [1, 2]), pl.Series("b", [3, 4])]):
        assert_series_equal(s1, s2)


def test_from_dict_struct() -> None:
    data: dict[str, dict[str, list[int]] | list[int]] = {
        "a": {"b": [1, 3], "c": [2, 4]},
        "d": [5, 6],
    }
    df = pl.from_dict(data)
    assert df.shape == (2, 2)
    assert df["a"][0] == {"b": 1, "c": 2}
    assert df["a"][1] == {"b": 3, "c": 4}
    assert df.schema == {"a": pl.Struct, "d": pl.Int64}


def test_from_dicts() -> None:
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": None}]
    df = pl.from_dicts(data)  # type: ignore[arg-type]
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, None)]
    assert df.schema == {"a": pl.Int64, "b": pl.Int64}


def test_from_dict_no_inference() -> None:
    schema = {"a": pl.Utf8}
    data = [{"a": "aa"}]
    pl.from_dicts(data, schema_overrides=schema, infer_schema_length=0)


def test_from_dicts_schema_override() -> None:
    schema = {
        "a": pl.Utf8,
        "b": pl.Int64,
        "c": pl.List(pl.Struct({"x": pl.Int64, "y": pl.Utf8, "z": pl.Float64})),
    }

    # initial data matches the expected schema
    data1 = [
        {
            "a": "l",
            "b": i,
            "c": [{"x": (j + 2), "y": "?", "z": (j % 2)} for j in range(2)],
        }
        for i in range(5)
    ]

    # extend with a mix of fields that are/not in the schema
    data2 = [{"b": i + 5, "d": "ABC", "e": "DEF"} for i in range(5)]

    for n_infer in (0, 3, 5, 8, 10, 100):
        df = pl.DataFrame(
            data=(data1 + data2),
            schema=schema,  # type: ignore[arg-type]
            infer_schema_length=n_infer,
        )
        assert df.schema == schema
        assert df.rows() == [
            ("l", 0, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 1, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 2, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 3, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            ("l", 4, [{"x": 2, "y": "?", "z": 0.0}, {"x": 3, "y": "?", "z": 1.0}]),
            (None, 5, None),
            (None, 6, None),
            (None, 7, None),
            (None, 8, None),
            (None, 9, None),
        ]


def test_from_dicts_struct() -> None:
    data = [{"a": {"b": 1, "c": 2}, "d": 5}, {"a": {"b": 3, "c": 4}, "d": 6}]
    df = pl.from_dicts(data)
    assert df.shape == (2, 2)
    assert df["a"][0] == {"b": 1, "c": 2}
    assert df["a"][1] == {"b": 3, "c": 4}

    # 5649
    assert pl.from_dicts([{"a": [{"x": 1}]}, {"a": [{"y": 1}]}]).to_dict(False) == {
        "a": [[{"y": None, "x": 1}], [{"y": 1, "x": None}]]
    }
    assert pl.from_dicts([{"a": [{"x": 1}, {"y": 2}]}, {"a": [{"y": 1}]}]).to_dict(
        False
    ) == {"a": [[{"y": None, "x": 1}, {"y": 2, "x": None}], [{"y": 1, "x": None}]]}


def test_from_records() -> None:
    data = [[1, 2, 3], [4, 5, 6]]
    df = pl.from_records(data, schema=["a", "b"])
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]


def test_from_numpy() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    df = pl.from_numpy(
        data,
        schema=["a", "b"],
        orient="col",
        schema_overrides={"a": pl.UInt32, "b": pl.UInt32},
    )
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]
    assert df.schema == {"a": pl.UInt32, "b": pl.UInt32}


def test_from_arrow() -> None:
    data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = pl.from_arrow(data)
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]  # type: ignore[union-attr]

    # if not a PyArrow type, raise a ValueError
    with pytest.raises(ValueError):
        _ = pl.from_arrow([1, 2])

    df = pl.from_arrow(
        data, schema=["a", "b"], schema_overrides={"a": pl.UInt32, "b": pl.UInt64}
    )
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]  # type: ignore[union-attr]
    assert df.schema == {"a": pl.UInt32, "b": pl.UInt64}  # type: ignore[union-attr]


def test_from_pandas_dataframe() -> None:
    pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    df = pl.from_pandas(pd_df)
    assert df.shape == (2, 3)
    assert df.rows() == [(1, 2, 3), (4, 5, 6)]

    # if not a pandas dataframe, raise a ValueError
    with pytest.raises(ValueError):
        _ = pl.from_pandas([1, 2])  # type: ignore[call-overload]


def test_from_pandas_series() -> None:
    pd_series = pd.Series([1, 2, 3], name="pd")
    s = pl.from_pandas(pd_series)
    assert s.shape == (3,)
    assert list(s) == [1, 2, 3]


def test_from_optional_not_available() -> None:
    from polars.dependencies import _LazyModule

    # proxy module is created dynamically if the required module is not available
    # (see the polars.dependencies source code for additional detail/comments)

    np = _LazyModule("numpy", module_available=False)
    with pytest.raises(ImportError, match=r"np\.array requires 'numpy'"):
        pl.from_numpy(np.array([[1, 2], [3, 4]]), schema=["a", "b"])

    pa = _LazyModule("pyarrow", module_available=False)
    with pytest.raises(ImportError, match=r"pa\.table requires 'pyarrow'"):
        pl.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))

    pd = _LazyModule("pandas", module_available=False)
    with pytest.raises(ImportError, match=r"pd\.Series requires 'pandas'"):
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
    out = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert out.shape == (128, 1)
    assert out["col_name"][0] == "value_0"
    assert out["col_name"][127] == "value_127"


def test_no_rechunk() -> None:
    table = pa.Table.from_pydict({"x": pa.chunked_array([list("ab"), list("cd")])})
    # table
    assert pl.from_arrow(table, rechunk=False).n_chunks() == 2
    # chunked array
    assert pl.from_arrow(table["x"], rechunk=False).n_chunks() == 2


def test_cat_to_pandas() -> None:
    df = pl.DataFrame({"a": ["best", "test"]})
    df = df.with_columns(pl.all().cast(pl.Categorical))
    pd_out = df.to_pandas()
    assert "category" in str(pd_out["a"].dtype)

    try:
        pd_pa_out = df.to_pandas(use_pyarrow_extension_array=True)
        assert pd_pa_out["a"].dtype.type == pa.DictionaryType
    except ModuleNotFoundError:
        # Skip test if Pandas 1.5.x is not installed.
        pass


def test_to_pandas() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [6, None, 8], "c": ["a", "b", "c"], "d": [None, "e", "f"]}
    )
    df = df.with_columns(
        [
            pl.col("c").cast(pl.Categorical).alias("e"),
            pl.col("d").cast(pl.Categorical).alias("f"),
        ]
    )
    pd_out = df.to_pandas()
    pd_out_dtypes_expected = [
        np.int64,
        np.float64,
        np.object_,
        np.object_,
        pd.CategoricalDtype(categories=["a", "b", "c"], ordered=False),
        pd.CategoricalDtype(categories=["e", "f"], ordered=False),
    ]
    assert pd_out.dtypes.to_list() == pd_out_dtypes_expected

    try:
        pd_pa_out = df.to_pandas(use_pyarrow_extension_array=True)
        pd_pa_dtypes_names = [dtype.name for dtype in pd_pa_out.dtypes]
        pd_pa_dtypes_names_expected = [
            "int64[pyarrow]",
            "int64[pyarrow]",
            "large_string[pyarrow]",
            "large_string[pyarrow]",
            "dictionary<values=large_string, indices=int64, ordered=0>[pyarrow]",
            "dictionary<values=large_string, indices=int64, ordered=0>[pyarrow]",
        ]
        assert pd_pa_dtypes_names == pd_pa_dtypes_names_expected
    except ModuleNotFoundError:
        # Skip test if Pandas 1.5.x is not installed.
        pass


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


def test_from_empty_pandas_with_dtypes() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    df["a"] = df["a"].astype(str)
    df["b"] = df["b"].astype(float)
    assert pl.from_pandas(df).dtypes == [pl.Utf8, pl.Float64]

    df = pl.DataFrame(
        data=[],
        schema={
            "a": pl.Int32,
            "b": pl.Datetime,
            "c": pl.Float32,
            "d": pl.Duration,
            "e": pl.Utf8,
        },
    ).to_pandas()
    assert pl.from_pandas(df).dtypes == [
        pl.Int32,
        pl.Datetime,
        pl.Float32,
        pl.Duration,
        pl.Utf8,
    ]


def test_from_empty_arrow() -> None:
    df = cast(pl.DataFrame, pl.from_arrow(pa.table(pd.DataFrame({"a": [], "b": []}))))
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Float64, pl.Float64]

    # 2705
    df1 = pd.DataFrame(columns=["b"], dtype=float)
    tbl = pa.Table.from_pandas(df1)
    out = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert out.columns == ["b", "__index_level_0__"]
    assert out.dtypes == [pl.Float64, pl.Null]
    tbl = pa.Table.from_pandas(df1, preserve_index=False)
    out = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert out.columns == ["b"]
    assert out.dtypes == [pl.Float64]

    # 4568
    tbl = pa.table({"l": []}, schema=pa.schema([("l", pa.large_list(pa.uint8()))]))

    df = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert df.schema["l"] == pl.List(pl.UInt8)


def test_from_null_column() -> None:
    df = pl.from_pandas(pd.DataFrame(data=[pd.NA, pd.NA], columns=["n/a"]))

    assert df.shape == (2, 1)
    assert df.columns == ["n/a"]
    assert dtype_to_py_type(df.dtypes[0]) is None


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
            assert_series_equal(s, pl.Series(["a", "a", "b"]).cast(pl.Categorical))


def test_from_pyarrow_chunked_array() -> None:
    column = pa.chunked_array([[1], [2]])
    series = pl.Series("column", column)
    assert series.to_list() == [1, 2]


def test_numpy_preserve_uint64_4112() -> None:
    assert pl.DataFrame({"a": [1, 2, 3]}).with_columns(
        pl.col("a").hash()
    ).to_numpy().dtype == np.dtype("uint64")


def test_view_ub() -> None:
    # this would be UB if the series was dropped and not passed to the view
    assert np.sum(pl.Series([3, 1, 5]).sort().view()) == 9


def test_arrow_list_null_5697() -> None:
    # Create a pyarrow table with a list[null] column.
    pa_table = pa.table([[[None]]], names=["mycol"])
    df = pl.from_arrow(pa_table)
    pa_table = df.to_arrow()
    # again to polars to test the schema
    assert pl.from_arrow(pa_table).schema == {  # type: ignore[union-attr]
        "mycol": pl.List(pl.Null)
    }


def test_from_pandas_null_struct_6412() -> None:
    data = [
        {
            "a": {
                "b": None,
            },
        },
        {"a": None},
    ]
    df_pandas = pd.DataFrame(data)
    assert pl.from_pandas(df_pandas).to_dict(False) == {"a": [{"b": None}, {"b": None}]}


@typing.no_type_check
def test_from_pyarrow_map() -> None:
    pa_table = pa.table(
        [[1, 2], [[("a", "something")], [("a", "else"), ("b", "another key")]]],
        schema=pa.schema(
            [("idx", pa.int16()), ("mapping", pa.map_(pa.string(), pa.string()))]
        ),
    )

    df = pl.from_arrow(pa_table)
    assert df.to_dict(False) == {
        "idx": [1, 2],
        "mapping": [
            [{"key": "a", "value": "something"}],
            [{"key": "a", "value": "else"}, {"key": "b", "value": "another key"}],
        ],
    }


def test_to_numpy_datelike() -> None:
    s = pl.Series(
        "dt",
        [
            datetime(2022, 7, 5, 10, 30, 45, 123456),
            None,
            datetime(2023, 2, 5, 15, 22, 30, 987654),
        ],
    )
    assert str(s.to_numpy()) == str(
        np.array(
            ["2022-07-05T10:30:45.123456", "NaT", "2023-02-05T15:22:30.987654"],
            dtype="datetime64[us]",
        )
    )
    assert str(s.drop_nulls().to_numpy()) == str(
        np.array(
            ["2022-07-05T10:30:45.123456", "2023-02-05T15:22:30.987654"],
            dtype="datetime64[us]",
        )
    )
