from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.fixture(
    params=[
        ("int8", [1, 3, 2], pl.Int8, np.int8),
        ("int16", [1, 3, 2], pl.Int16, np.int16),
        ("int32", [1, 3, 2], pl.Int32, np.int32),
        ("int64", [1, 3, 2], pl.Int64, np.int64),
        ("uint8", [1, 3, 2], pl.UInt8, np.uint8),
        ("uint16", [1, 3, 2], pl.UInt16, np.uint16),
        ("uint32", [1, 3, 2], pl.UInt32, np.uint32),
        ("uint64", [1, 3, 2], pl.UInt64, np.uint64),
        ("float32", [21.7, 21.8, 21], pl.Float32, np.float32),
        ("float64", [21.7, 21.8, 21], pl.Float64, np.float64),
        ("bool", [True, False, False], pl.Boolean, np.bool_),
        ("object", [21.7, "string1", object()], pl.Object, np.object_),
        ("str", ["string1", "string2", "string3"], pl.Utf8, np.str_),
        ("intc", [1, 3, 2], pl.Int32, np.intc),
        ("uintc", [1, 3, 2], pl.UInt32, np.uintc),
        ("str_fixed", ["string1", "string2", "string3"], pl.Utf8, np.str_),
        (
            "bytes",
            [b"byte_string1", b"byte_string2", b"byte_string3"],
            pl.Binary,
            np.bytes_,
        ),
    ]
)
def numpy_interop_test_data(request: Any) -> Any:
    return request.param


def test_df_from_numpy(numpy_interop_test_data: Any) -> None:
    name, values, pl_dtype, np_dtype = numpy_interop_test_data
    df = pl.DataFrame({name: np.array(values, dtype=np_dtype)})
    assert [pl_dtype] == df.dtypes


def test_asarray(numpy_interop_test_data: Any) -> None:
    name, values, pl_dtype, np_dtype = numpy_interop_test_data
    pl_series_to_numpy_array = np.asarray(pl.Series(name, values, pl_dtype))
    numpy_array = np.asarray(values, dtype=np_dtype)
    assert_array_equal(pl_series_to_numpy_array, numpy_array)


@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_to_numpy(numpy_interop_test_data: Any, use_pyarrow: bool) -> None:
    name, values, pl_dtype, np_dtype = numpy_interop_test_data
    pl_series_to_numpy_array = pl.Series(name, values, pl_dtype).to_numpy(
        use_pyarrow=use_pyarrow
    )
    numpy_array = np.asarray(values, dtype=np_dtype)
    assert_array_equal(pl_series_to_numpy_array, numpy_array)


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.parametrize("has_null", [True, False])
@pytest.mark.parametrize("dtype", [pl.Time, pl.Boolean, pl.Utf8])
def test_to_numpy_no_zero_copy(
    use_pyarrow: bool, has_null: bool, dtype: pl.PolarsDataType
) -> None:
    data: list[Any] = ["a", None] if dtype == pl.Utf8 else [0, None]
    series = pl.Series(data if has_null else data[:1], dtype=dtype)
    with pytest.raises(ValueError):
        series.to_numpy(zero_copy_only=True, use_pyarrow=use_pyarrow)


def test_to_numpy_empty_no_pyarrow() -> None:
    series = pl.Series([], dtype=pl.Null)
    result = series.to_numpy()
    assert result.dtype == pl.Float32
    assert result.shape == (0,)
    assert result.size == 0


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


@pytest.mark.parametrize(
    "nulls",
    [
        [],
        [None],
        [None, None],
        [None, None, None],
    ],
)
def test_from_pandas_nulls(nulls: list[None]) -> None:
    # empty and/or all null values, no pandas dtype
    ps = pd.Series(nulls)
    s = pl.from_pandas(ps)
    assert nulls == s.to_list()


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
    assert df.to_dict(as_series=False) == {
        "val": [100, 200, 300],
        "misc": ["x", "y", "z"],
    }

    df = pl.from_pandas(pd_df.set_index(["dtm", "val"]))
    assert df.to_dict(as_series=False) == {"misc": ["x", "y", "z"]}

    df = pl.from_pandas(pd_df.set_index(["dtm"]), include_index=True)
    assert df.to_dict(as_series=False) == data

    df = pl.from_pandas(pd_df.set_index(["dtm", "val"]), include_index=True)
    assert df.to_dict(as_series=False) == data


def test_from_pandas_duplicated_columns() -> None:
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["a", "b", "c", "b"])
    with pytest.raises(ValueError, match="duplicate column names found: "):
        pl.from_pandas(df)


def test_arrow_list_roundtrip() -> None:
    # https://github.com/pola-rs/polars/issues/1064
    tbl = pa.table({"a": [1], "b": [[1, 2]]})
    arw = pl.from_arrow(tbl).to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_null_roundtrip() -> None:
    tbl = pa.table({"a": [None, None], "b": [[None, None], [None, None]]})
    df = pl.from_arrow(tbl)

    if isinstance(df, pl.DataFrame):
        assert df.dtypes == [pl.Null, pl.List(pl.Null)]

    arw = df.to_arrow()

    assert arw.shape == tbl.shape
    assert arw.schema.names == tbl.schema.names
    for c1, c2 in zip(arw.columns, tbl.columns):
        assert c1.to_pylist() == c2.to_pylist()


def test_arrow_empty_dataframe() -> None:
    # 0x0 dataframe
    df = pl.DataFrame({})
    tbl = pa.table({})
    assert df.to_arrow() == tbl
    df2 = cast(pl.DataFrame, pl.from_arrow(df.to_arrow()))
    assert_frame_equal(df2, df)

    # 0 row dataframe
    df = pl.DataFrame({}, schema={"a": pl.Int32})
    tbl = pa.Table.from_batches([], pa.schema([pa.field("a", pa.int32())]))
    assert df.to_arrow() == tbl
    df2 = cast(pl.DataFrame, pl.from_arrow(df.to_arrow()))
    assert df2.schema == {"a": pl.Int32}
    assert df2.shape == (0, 1)


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
    assert pl.from_dicts([{"a": [{"x": 1}]}, {"a": [{"y": 1}]}]).to_dict(
        as_series=False
    ) == {"a": [[{"y": None, "x": 1}], [{"y": 1, "x": None}]]}
    assert pl.from_dicts([{"a": [{"x": 1}, {"y": 2}]}, {"a": [{"y": 1}]}]).to_dict(
        as_series=False
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


def test_from_numpy_structured() -> None:
    test_data = [
        ("Google Pixel 7", 521.90, True),
        ("Apple iPhone 14 Pro", 999.00, True),
        ("Samsung Galaxy S23 Ultra", 1199.99, False),
        ("OnePlus 11", 699.00, True),
    ]
    # create a numpy structured array...
    arr_structured = np.array(
        test_data,
        dtype=np.dtype(
            [
                ("product", "U32"),
                ("price_usd", "float64"),
                ("in_stock", "bool"),
            ]
        ),
    )
    # ...and also establish as a record array view
    arr_records = arr_structured.view(np.recarray)

    # confirm that we can cleanly initialise a DataFrame from both,
    # respecting the native dtypes and any schema overrides, etc.
    for arr in (arr_structured, arr_records):
        df = pl.DataFrame(data=arr).sort(by="price_usd", descending=True)

        assert df.schema == {
            "product": pl.Utf8,
            "price_usd": pl.Float64,
            "in_stock": pl.Boolean,
        }
        assert df.rows() == sorted(test_data, key=lambda row: -row[1])

        for df in (
            pl.DataFrame(
                data=arr, schema=["phone", ("price_usd", pl.Float32), "available"]
            ),
            pl.DataFrame(
                data=arr,
                schema=["phone", "price_usd", "available"],
                schema_overrides={"price_usd": pl.Float32},
            ),
        ):
            assert df.schema == {
                "phone": pl.Utf8,
                "price_usd": pl.Float32,
                "available": pl.Boolean,
            }


def test_from_arrow() -> None:
    data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = pl.from_arrow(data)
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 4), (2, 5), (3, 6)]  # type: ignore[union-attr]

    # if not a PyArrow type, raise a TypeError
    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
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
    # https://github.com/pola-rs/polars/issues/1752
    tbls = [
        pa.table(
            {
                "col_name": pa.array(
                    [f"value_{i}"], pa.dictionary(pa.int8(), pa.string())
                )
            }
        )
        for i in range(128)
    ]

    tbl = pa.concat_tables(tbls, promote_options="default")
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
    assert isinstance(pd_out["a"].dtype, pd.CategoricalDtype)

    pd_pa_out = df.to_pandas(use_pyarrow_extension_array=True)
    assert pd_pa_out["a"].dtype == pd.ArrowDtype(
        pa.dictionary(pa.int64(), pa.large_string())
    )


def test_to_pandas() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [6, None, 8],
            "c": [10.0, 25.0, 50.5],
            "d": [date(2023, 7, 5), None, date(1999, 12, 13)],
            "e": ["a", "b", "c"],
            "f": [None, "e", "f"],
            "g": [datetime.now(), datetime.now(), None],
        },
        schema_overrides={"a": pl.UInt8},
    ).with_columns(
        [
            pl.col("e").cast(pl.Categorical).alias("h"),
            pl.col("f").cast(pl.Categorical).alias("i"),
        ]
    )

    pd_out = df.to_pandas()
    ns_datetimes = pa.__version__ < "13"

    pd_out_dtypes_expected = [
        np.dtype(np.uint8),
        np.dtype(np.float64),
        np.dtype(np.float64),
        np.dtype(f"datetime64[{'ns' if ns_datetimes else 'ms'}]"),
        np.dtype(np.object_),
        np.dtype(np.object_),
        np.dtype(f"datetime64[{'ns' if ns_datetimes else 'us'}]"),
        pd.CategoricalDtype(categories=["a", "b", "c"], ordered=False),
        pd.CategoricalDtype(categories=["e", "f"], ordered=False),
    ]
    assert pd_out_dtypes_expected == pd_out.dtypes.to_list()

    pd_out_dtypes_expected[3] = np.dtype("O")
    pd_out = df.to_pandas(date_as_object=True)
    assert pd_out_dtypes_expected == pd_out.dtypes.to_list()

    try:
        pd_pa_out = df.to_pandas(use_pyarrow_extension_array=True)
        pd_pa_dtypes_names = [dtype.name for dtype in pd_pa_out.dtypes]
        pd_pa_dtypes_names_expected = [
            "uint8[pyarrow]",
            "int64[pyarrow]",
            "double[pyarrow]",
            "date32[day][pyarrow]",
            "large_string[pyarrow]",
            "large_string[pyarrow]",
            "timestamp[us][pyarrow]",
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
    df1 = pd.DataFrame(columns=["b"], dtype=float, index=pd.Index([]))
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
    assert df.dtypes[0] == pl.Null


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


def test_pandas_string_none_conversion_3298() -> None:
    data: dict[str, list[str | None]] = {"col_1": ["a", "b", "c", "d"]}
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
            assert_series_equal(
                s, pl.Series(["a", "a", "b"]).cast(pl.Categorical), check_names=False
            )


def test_from_pyarrow_chunked_array() -> None:
    column = pa.chunked_array([[1], [2]])
    series = pl.Series("column", column)
    assert series.to_list() == [1, 2]


def test_numpy_preserve_uint64_4112() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}).with_columns(pl.col("a").hash())
    assert df.to_numpy().dtype == np.dtype("uint64")
    assert df.to_numpy(structured=True).dtype == np.dtype([("a", "uint64")])


def test_arrow_list_null_5697() -> None:
    # Create a pyarrow table with a list[null] column.
    pa_table = pa.table([[[None]]], names=["mycol"])
    df = pl.from_arrow(pa_table)
    pa_table = df.to_arrow()
    # again to polars to test the schema
    assert pl.from_arrow(pa_table).schema == {"mycol": pl.List(pl.Null)}  # type: ignore[union-attr]


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
    assert pl.from_pandas(df_pandas).to_dict(as_series=False) == {
        "a": [{"b": None}, {"b": None}]
    }


def test_from_pyarrow_map() -> None:
    pa_table = pa.table(
        [[1, 2], [[("a", "something")], [("a", "else"), ("b", "another key")]]],
        schema=pa.schema(
            [("idx", pa.int16()), ("mapping", pa.map_(pa.string(), pa.string()))]
        ),
    )

    result = cast(pl.DataFrame, pl.from_arrow(pa_table))
    assert result.to_dict(as_series=False) == {
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


def test_from_fixed_size_binary_list() -> None:
    val = [[b"63A0B1C66575DD5708E1EB2B"]]
    arrow_array = pa.array(val, type=pa.list_(pa.binary(24)))
    s = cast(pl.Series, pl.from_arrow(arrow_array))
    assert s.dtype == pl.List(pl.Binary)
    assert s.to_list() == val


def test_dataframe_from_repr() -> None:
    # round-trip various types
    with pl.StringCache():
        frame = (
            pl.LazyFrame(
                {
                    "a": [1, 2, None],
                    "b": [4.5, 5.5, 6.5],
                    "c": ["x", "y", "z"],
                    "d": [True, False, True],
                    "e": [None, "", None],
                    "f": [date(2022, 7, 5), date(2023, 2, 5), date(2023, 8, 5)],
                    "g": [time(0, 0, 0, 1), time(12, 30, 45), time(23, 59, 59, 999000)],
                    "h": [
                        datetime(2022, 7, 5, 10, 30, 45, 4560),
                        datetime(2023, 10, 12, 20, 3, 8, 11),
                        None,
                    ],
                },
            )
            .with_columns(
                pl.col("c").cast(pl.Categorical),
                pl.col("h").cast(pl.Datetime("ns")),
            )
            .collect()
        )

        assert frame.schema == {
            "a": pl.Int64,
            "b": pl.Float64,
            "c": pl.Categorical,
            "d": pl.Boolean,
            "e": pl.Utf8,
            "f": pl.Date,
            "g": pl.Time,
            "h": pl.Datetime("ns"),
        }
        df = cast(pl.DataFrame, pl.from_repr(repr(frame)))
        assert_frame_equal(frame, df)

    # empty frame; confirm schema is inferred
    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
        ┌─────┬─────┬─────┬─────┬─────┬───────┐
        │ id  ┆ q1  ┆ q2  ┆ q3  ┆ q4  ┆ total │
        │ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i8  ┆ i16 ┆ i32 ┆ i64 ┆ f64   │
        ╞═════╪═════╪═════╪═════╪═════╪═══════╡
        └─────┴─────┴─────┴─────┴─────┴───────┘
        """
        ),
    )
    assert df.shape == (0, 6)
    assert df.rows() == []
    assert df.schema == {
        "id": pl.Utf8,
        "q1": pl.Int8,
        "q2": pl.Int16,
        "q3": pl.Int32,
        "q4": pl.Int64,
        "total": pl.Float64,
    }

    # empty frame with no dtypes
    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
        ┌──────┬───────┐
        │ misc ┆ other │
        ╞══════╪═══════╡
        └──────┴───────┘
        """
        ),
    )
    assert_frame_equal(df, pl.DataFrame(schema={"misc": pl.Utf8, "other": pl.Utf8}))

    # empty frame with non-standard/blank 'null'
    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
            ┌─────┬─────┐
            │ c1  ┆ c2  │
            │ --- ┆ --- │
            │ i32 ┆ f64 │
            ╞═════╪═════╡
            │     │     │
            └─────┴─────┘
            """
        ),
    )
    assert_frame_equal(
        df,
        pl.DataFrame(data=[(None, None)], schema={"c1": pl.Int32, "c2": pl.Float64}),
    )

    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
        # >>> Missing cols with old-style ellipsis, nulls, commented out
        # ┌────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────┐
        # │ dt         ┆ c1  ┆ c2  ┆ c3  ┆ ... ┆ c96 ┆ c97 ┆ c98 ┆ c99  │
        # │ ---        ┆ --- ┆ --- ┆ --- ┆     ┆ --- ┆ --- ┆ --- ┆ ---  │
        # │ date       ┆ i32 ┆ i32 ┆ i32 ┆     ┆ i64 ┆ i64 ┆ i64 ┆ i64  │
        # ╞════════════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╪══════╡
        # │ 2023-03-25 ┆ 1   ┆ 2   ┆ 3   ┆ ... ┆ 96  ┆ 97  ┆ 98  ┆ 99   │
        # │ 1999-12-31 ┆ 3   ┆ 6   ┆ 9   ┆ ... ┆ 288 ┆ 291 ┆ 294 ┆ null │
        # │ null       ┆ 9   ┆ 18  ┆ 27  ┆ ... ┆ 864 ┆ 873 ┆ 882 ┆ 891  │
        # └────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────┘
        """
        ),
    )
    assert df.schema == {
        "dt": pl.Date,
        "c1": pl.Int32,
        "c2": pl.Int32,
        "c3": pl.Int32,
        "c96": pl.Int64,
        "c97": pl.Int64,
        "c98": pl.Int64,
        "c99": pl.Int64,
    }
    assert df.rows() == [
        (date(2023, 3, 25), 1, 2, 3, 96, 97, 98, 99),
        (date(1999, 12, 31), 3, 6, 9, 288, 291, 294, None),
        (None, 9, 18, 27, 864, 873, 882, 891),
    ]

    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
        # >>> no dtypes:
        # ┌────────────┬──────┐
        # │ dt         ┆ c99  │
        # ╞════════════╪══════╡
        # │ 2023-03-25 ┆ 99   │
        # │ 1999-12-31 ┆ null │
        # │ null       ┆ 891  │
        # └────────────┴──────┘
        """
        ),
    )
    assert df.schema == {"dt": pl.Date, "c99": pl.Int64}
    assert df.rows() == [
        (date(2023, 3, 25), 99),
        (date(1999, 12, 31), None),
        (None, 891),
    ]

    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
        In [2]: with pl.Config() as cfg:
           ...:     pl.Config.set_tbl_formatting("UTF8_FULL", rounded_corners=True)
           ...:     print(df)
           ...:
        shape: (1, 5)
        ╭───────────┬────────────┬───┬───────┬────────────────────────────────╮
        │ source_ac ┆ source_cha ┆ … ┆ ident ┆ timestamp                      │
        │ tor_id    ┆ nnel_id    ┆   ┆ ---   ┆ ---                            │
        │ ---       ┆ ---        ┆   ┆ str   ┆ datetime[μs, Asia/Tokyo]       │
        │ i32       ┆ i64        ┆   ┆       ┆                                │
        ╞═══════════╪════════════╪═══╪═══════╪════════════════════════════════╡
        │ 123456780 ┆ 9876543210 ┆ … ┆ a:b:c ┆ 2023-03-25 10:56:59.663053 JST │
        ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ …         ┆ …          ┆ … ┆ …     ┆ …                              │
        ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 803065983 ┆ 2055938745 ┆ … ┆ x:y:z ┆ 2023-03-25 12:38:18.050545 JST │
        ╰───────────┴────────────┴───┴───────┴────────────────────────────────╯
        # "Een fluitje van een cent..." :)
        """
        ),
    )
    assert df.shape == (2, 4)
    assert df.schema == {
        "source_actor_id": pl.Int32,
        "source_channel_id": pl.Int64,
        "ident": pl.Utf8,
        "timestamp": pl.Datetime("us", "Asia/Tokyo"),
    }


def test_series_from_repr() -> None:
    with pl.StringCache():
        frame = (
            pl.LazyFrame(
                {
                    "a": [1, 2, None],
                    "b": [4.5, 5.5, 6.5],
                    "c": ["x", "y", "z"],
                    "d": [True, False, True],
                    "e": [None, "", None],
                    "f": [date(2022, 7, 5), date(2023, 2, 5), date(2023, 8, 5)],
                    "g": [time(0, 0, 0, 1), time(12, 30, 45), time(23, 59, 59, 999000)],
                    "h": [
                        datetime(2022, 7, 5, 10, 30, 45, 4560),
                        datetime(2023, 10, 12, 20, 3, 8, 11),
                        None,
                    ],
                },
            )
            .with_columns(
                pl.col("c").cast(pl.Categorical),
                pl.col("h").cast(pl.Datetime("ns")),
            )
            .collect()
        )

        for col in frame.columns:
            s = cast(pl.Series, pl.from_repr(repr(frame[col])))
            assert_series_equal(s, frame[col])

    s = cast(
        pl.Series,
        pl.from_repr(
            """
            Out[3]:
            shape: (3,)
            Series: 's' [str]
            [
                "a"
                 …
                "c"
            ]
            """
        ),
    )
    assert_series_equal(s, pl.Series("s", ["a", "c"]))

    s = cast(
        pl.Series,
        pl.from_repr(
            """
            Series: 'flt' [f32]
            [
            ]
            """
        ),
    )
    assert_series_equal(s, pl.Series("flt", [], dtype=pl.Float32))


def test_dataframe_from_repr_custom_separators() -> None:
    # repr created with custom digit-grouping
    # and non-default group/decimal separators
    df = cast(
        pl.DataFrame,
        pl.from_repr(
            """
            ┌───────────┬────────────┐
            │ x         ┆ y          │
            │ ---       ┆ ---        │
            │ i32       ┆ f64        │
            ╞═══════════╪════════════╡
            │ 123.456   ┆ -10.000,55 │
            │ -9.876    ┆ 10,0       │
            │ 9.999.999 ┆ 8,5e8      │
            └───────────┴────────────┘
            """
        ),
    )
    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "x": [123456, -9876, 9999999],
                "y": [-10000.55, 10.0, 850000000.0],
            },
            schema={"x": pl.Int32, "y": pl.Float64},
        ),
    )


def test_to_init_repr() -> None:
    # round-trip various types
    with pl.StringCache():
        df = (
            pl.LazyFrame(
                {
                    "a": [1, 2, None],
                    "b": [4.5, 5.5, 6.5],
                    "c": ["x", "y", "z"],
                    "d": [True, False, True],
                    "e": [None, "", None],
                    "f": [date(2022, 7, 5), date(2023, 2, 5), date(2023, 8, 5)],
                    "g": [time(0, 0, 0, 1), time(12, 30, 45), time(23, 59, 59, 999000)],
                    "h": [
                        datetime(2022, 7, 5, 10, 30, 45, 4560),
                        datetime(2023, 10, 12, 20, 3, 8, 11),
                        None,
                    ],
                },
            )
            .with_columns(
                pl.col("c").cast(pl.Categorical),
                pl.col("h").cast(pl.Datetime("ns")),
            )
            .collect()
        )

        assert_frame_equal(eval(df.to_init_repr().replace("datetime.", "")), df)


def test_untrusted_categorical_input() -> None:
    df = pd.DataFrame({"x": pd.Categorical(["x"], ["x", "y"])})
    assert pl.from_pandas(df).group_by("x").count().to_dict(as_series=False) == {
        "x": ["x"],
        "count": [1],
    }


def test_sliced_struct_from_arrow() -> None:
    # Create a dataset with 3 rows
    tbl = pa.Table.from_arrays(
        arrays=[
            pa.StructArray.from_arrays(
                arrays=[
                    pa.array([1, 2, 3], pa.int32()),
                    pa.array(["foo", "bar", "baz"], pa.utf8()),
                ],
                names=["a", "b"],
            )
        ],
        names=["struct_col"],
    )

    # slice the table
    # check if FFI correctly reads sliced
    result = cast(pl.DataFrame, pl.from_arrow(tbl.slice(1, 2)))
    assert result.to_dict(as_series=False) == {
        "struct_col": [{"a": 2, "b": "bar"}, {"a": 3, "b": "baz"}]
    }

    result = cast(pl.DataFrame, pl.from_arrow(tbl.slice(1, 1)))
    assert result.to_dict(as_series=False) == {"struct_col": [{"a": 2, "b": "bar"}]}


def test_from_arrow_invalid_time_zone() -> None:
    arr = pa.array(
        [datetime(2021, 1, 1, 0, 0, 0, 0)], type=pa.timestamp("ns", tz="+01:00")
    )
    with pytest.raises(ComputeError, match=r"unable to parse time zone: '\+01:00'"):
        pl.from_arrow(arr)


def test_from_avro_invalid_time_zone_13032() -> None:
    arr = pa.array(
        [datetime(2021, 1, 1, 0, 0, 0, 0)], type=pa.timestamp("ns", tz="00:00")
    )
    result = cast(pl.Series, pl.from_arrow(arr))
    expected = pl.Series([datetime(2021, 1, 1)], dtype=pl.Datetime("ns", "UTC"))
    assert_series_equal(result, expected)
