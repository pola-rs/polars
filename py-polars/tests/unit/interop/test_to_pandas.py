from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa

import polars as pl


def test_df_to_pandas_empty() -> None:
    df = pl.DataFrame()
    result = df.to_pandas()
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)


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

    pd_out_dtypes_expected = [
        np.dtype(np.uint8),
        np.dtype(np.float64),
        np.dtype(np.float64),
        np.dtype("datetime64[ms]"),
        np.dtype(np.object_),
        np.dtype(np.object_),
        np.dtype("datetime64[us]"),
        pd.CategoricalDtype(categories=["a", "b", "c"], ordered=False),
        pd.CategoricalDtype(categories=["e", "f"], ordered=False),
    ]
    assert pd_out_dtypes_expected == pd_out.dtypes.to_list()

    pd_out_dtypes_expected[3] = np.dtype("O")
    pd_out = df.to_pandas(date_as_object=True)
    assert pd_out_dtypes_expected == pd_out.dtypes.to_list()

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


def test_cat_to_pandas() -> None:
    df = pl.DataFrame({"a": ["best", "test"]})
    df = df.with_columns(pl.all().cast(pl.Categorical))

    pd_out = df.to_pandas()
    assert isinstance(pd_out["a"].dtype, pd.CategoricalDtype)

    pd_pa_out = df.to_pandas(use_pyarrow_extension_array=True)
    assert pd_pa_out["a"].dtype == pd.ArrowDtype(
        pa.dictionary(pa.int64(), pa.large_string())
    )


def test_from_empty_pandas_with_dtypes() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    df["a"] = df["a"].astype(str)
    df["b"] = df["b"].astype(float)
    assert pl.from_pandas(df).dtypes == [pl.String, pl.Float64]

    df = pl.DataFrame(
        data=[],
        schema={
            "a": pl.Int32,
            "b": pl.Datetime,
            "c": pl.Float32,
            "d": pl.Duration,
            "e": pl.String,
        },
    ).to_pandas()

    assert pl.from_pandas(df).dtypes == [
        pl.Int32,
        pl.Datetime,
        pl.Float32,
        pl.Duration,
        pl.String,
    ]


def test_to_pandas_series() -> None:
    assert (pl.Series("a", [1, 2, 3]).to_pandas() == pd.Series([1, 2, 3])).all()


def test_to_pandas_date() -> None:
    data = [date(1990, 1, 1), date(2024, 12, 31)]
    s = pl.Series("a", data)

    result_series = s.to_pandas()
    expected_series = pd.Series(data, dtype="datetime64[ms]", name="a")
    pd.testing.assert_series_equal(result_series, expected_series)

    result_df = s.to_frame().to_pandas()
    expected_df = expected_series.to_frame()
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_to_pandas_datetime() -> None:
    data = [datetime(1990, 1, 1, 0, 0, 0), datetime(2024, 12, 31, 23, 59, 59)]
    s = pl.Series("a", data)

    result_series = s.to_pandas()
    expected_series = pd.Series(data, dtype="datetime64[us]", name="a")
    pd.testing.assert_series_equal(result_series, expected_series)

    result_df = s.to_frame().to_pandas()
    expected_df = expected_series.to_frame()
    pd.testing.assert_frame_equal(result_df, expected_df)
