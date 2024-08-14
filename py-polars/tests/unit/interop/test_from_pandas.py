from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_index_not_silently_excluded() -> None:
    ddict = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = pd.DataFrame(ddict, index=pd.Index([7, 8, 9], name="a"))
    with pytest.raises(ValueError, match="indices and column names must not overlap"):
        pl.from_pandas(df, include_index=True)


def test_nameless_multiindex_doesnt_raise_with_include_index_false_18130() -> None:
    df = pd.DataFrame(
        range(4),
        columns=["A"],
        index=pd.MultiIndex.from_product((["C", "D"], [3, 4])),
    )
    result = pl.from_pandas(df)
    expected = pl.DataFrame({"A": [0, 1, 2, 3]})
    assert_frame_equal(result, expected)


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
        "strings": pl.String,
        "strings_nulls": pl.String,
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

    date_times = pd.date_range("2021-06-24 00:00:00", "2021-06-24 09:00:00", freq="1h")
    s = pl.from_pandas(date_times)
    assert s[0] == datetime(2021, 6, 24, 0, 0)
    assert s[-1] == datetime(2021, 6, 24, 9, 0)


@pytest.mark.parametrize(
    ("index_class", "index_data", "index_params", "expected_data", "expected_dtype"),
    [
        (pd.Index, [100, 200, 300], {}, None, pl.Int64),
        (pd.Index, [1, 2, 3], {"dtype": "uint32"}, None, pl.UInt32),
        (pd.RangeIndex, 5, {}, [0, 1, 2, 3, 4], pl.Int64),
        (pd.CategoricalIndex, ["N", "E", "S", "W"], {}, None, pl.Categorical),
        (
            pd.DatetimeIndex,
            [datetime(1960, 12, 31), datetime(2077, 10, 20)],
            {"dtype": "datetime64[ms]"},
            None,
            pl.Datetime("ms"),
        ),
        (
            pd.TimedeltaIndex,
            ["24 hours", "2 days 8 hours", "3 days 42 seconds"],
            {},
            [timedelta(1), timedelta(days=2, hours=8), timedelta(days=3, seconds=42)],
            pl.Duration("ns"),
        ),
    ],
)
def test_from_pandas_index(
    index_class: Any,
    index_data: Any,
    index_params: dict[str, Any],
    expected_data: list[Any] | None,
    expected_dtype: PolarsDataType,
) -> None:
    if expected_data is None:
        expected_data = index_data

    s = pl.from_pandas(index_class(index_data, **index_params))
    assert s.to_list() == expected_data
    assert s.dtype == expected_dtype


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


def test_duplicate_cols_diff_types() -> None:
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["0", 0, "1", 1])
    with pytest.raises(
        ValueError,
        match="Pandas dataframe contains non-unique indices and/or column names",
    ):
        pl.from_pandas(df)


def test_from_pandas_duplicated_columns() -> None:
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["a", "b", "c", "b"])
    with pytest.raises(
        ValueError,
        match="Pandas dataframe contains non-unique indices and/or column names",
    ):
        pl.from_pandas(df)


def test_from_pandas_null() -> None:
    # null column is an object dtype, so pl.Utf8 is most close
    df = pd.DataFrame([{"a": None}, {"a": None}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.String]
    assert out["a"][0] is None

    df = pd.DataFrame([{"a": None, "b": 1}, {"a": None, "b": 2}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.String, pl.Int64]


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


def test_from_null_column() -> None:
    df = pl.from_pandas(pd.DataFrame(data=[pd.NA, pd.NA], columns=["n/a"]))

    assert df.shape == (2, 1)
    assert df.columns == ["n/a"]
    assert df.dtypes[0] == pl.Null


def test_from_pandas_ns_resolution() -> None:
    df = pd.DataFrame(
        [pd.Timestamp(year=2021, month=1, day=1, hour=1, second=1, nanosecond=1)],
        columns=["date"],
    )
    assert pl.from_pandas(df)[0, 0] == datetime(2021, 1, 1, 1, 0, 1)


def test_pandas_string_none_conversion_3298() -> None:
    data: dict[str, list[str | None]] = {"col_1": ["a", "b", "c", "d"]}
    data["col_1"][0] = None
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(df_pd)
    assert df_pl.to_series().to_list() == [None, "b", "c", "d"]


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
        "a": [{"b": None}, None]
    }


def test_untrusted_categorical_input() -> None:
    df_pd = pd.DataFrame({"x": pd.Categorical(["x"], ["x", "y"])})
    df = pl.from_pandas(df_pd)
    result = df.group_by("x").len()
    expected = pl.DataFrame(
        {"x": ["x"], "len": [1]}, schema={"x": pl.Categorical, "len": pl.UInt32}
    )
    assert_frame_equal(result, expected, categorical_as_str=True)


@pytest.fixture()
def _set_pyarrow_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polars._utils.construction.dataframe._PYARROW_AVAILABLE", False
    )
    monkeypatch.setattr("polars._utils.construction.series._PYARROW_AVAILABLE", False)


@pytest.mark.usefixtures("_set_pyarrow_unavailable")
def test_from_pandas_pyarrow_not_available_succeeds() -> None:
    data: dict[str, Any] = {
        "a": [1, 2],
        "b": ["one", "two"],
        "c": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        "d": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[us]"),
        "e": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ms]"),
        "f": np.array([1, 2], dtype="timedelta64[ns]"),
        "g": np.array([1, 2], dtype="timedelta64[us]"),
        "h": np.array([1, 2], dtype="timedelta64[ms]"),
        "i": [True, False],
    }

    # DataFrame
    result = pl.from_pandas(pd.DataFrame(data))
    expected = pl.DataFrame(data)
    assert_frame_equal(result, expected)

    # Series
    for col in data:
        s_pd = pd.Series(data[col])
        result_s = pl.from_pandas(s_pd)
        expected_s = pl.Series(data[col])
        assert_series_equal(result_s, expected_s)


@pytest.mark.usefixtures("_set_pyarrow_unavailable")
def test_from_pandas_pyarrow_not_available_fails() -> None:
    with pytest.raises(ImportError, match="pyarrow is required"):
        pl.from_pandas(pd.DataFrame({"a": [1, 2, 3]}, dtype="Int64"))
    with pytest.raises(ImportError, match="pyarrow is required"):
        pl.from_pandas(pd.Series([1, 2, 3], dtype="Int64"))
    with pytest.raises(ImportError, match="pyarrow is required"):
        pl.from_pandas(
            pd.DataFrame({"a": pd.to_datetime(["2020-01-01T00:00+01:00"]).to_series()})
        )
    with pytest.raises(ImportError, match="pyarrow is required"):
        pl.from_pandas(pd.DataFrame({"a": [None, "foo"]}))


def test_from_pandas_nan_to_null_16453(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polars._utils.construction.dataframe._MIN_NUMPY_SIZE_FOR_MULTITHREADING", 2
    )
    df = pd.DataFrame(
        {"a": [np.nan, 1.0, 2], "b": [1.0, 2.0, 3.0], "c": [4.0, 5.0, 6.0]}
    )
    result = pl.from_pandas(df, nan_to_null=True)
    expected = pl.DataFrame(
        {"a": [None, 1.0, 2], "b": [1.0, 2.0, 3.0], "c": [4.0, 5.0, 6.0]}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("null", [pd.NA, np.nan, None, float("nan")])
def test_from_pandas_string_with_natype_17355(null: Any) -> None:
    # https://github.com/pola-rs/polars/issues/17355

    pd_df = pd.DataFrame({"col": ["a", null]})
    result = pl.from_pandas(pd_df)
    expected = pl.DataFrame({"col": ["a", None]})
    assert_frame_equal(result, expected)
