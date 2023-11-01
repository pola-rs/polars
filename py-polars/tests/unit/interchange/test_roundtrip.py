from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.interchange
import pytest
from hypothesis import given, note

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

protocol_dtypes = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Boolean,
    pl.Utf8,
    pl.Datetime,
    pl.Categorical,
]


@given(dataframes(allowed_dtypes=protocol_dtypes, excluded_dtypes=[pl.Boolean]))
def test_roundtrip_pyarrow_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__()
    note(f"n_chunks: {dfi.num_chunks()}")
    df_pa = pa.interchange.from_dataframe(dfi)
    with pl.StringCache():
        result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[pl.Boolean, pl.Categorical],
        chunked=False,
    )
)
def test_roundtrip_pyarrow_zero_copy_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__(allow_copy=False)
    note(f"n_chunks: {dfi.num_chunks()}")
    df_pa = pa.interchange.from_dataframe(dfi, allow_copy=False)
    result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]
    assert_frame_equal(result, df, categorical_as_str=True)


@given(dataframes(allowed_dtypes=protocol_dtypes))
@pytest.mark.filterwarnings(
    "ignore:.*PEP3118 format string that does not match its itemsize:RuntimeWarning"
)
def test_roundtrip_pandas_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__()
    df_pd = pd.api.interchange.from_dataframe(dfi)
    result = pl.from_pandas(df_pd, nan_to_null=False)
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[pl.Categorical],
        chunked=False,
    )
)
@pytest.mark.filterwarnings(
    "ignore:.*PEP3118 format string that does not match its itemsize:RuntimeWarning"
)
def test_roundtrip_pandas_zero_copy_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__(allow_copy=False)
    df_pd = pd.api.interchange.from_dataframe(dfi, allow_copy=False)
    result = pl.from_pandas(df_pd, nan_to_null=False)
    assert_frame_equal(result, df, categorical_as_str=True)


def test_roundtrip_pandas_boolean_subchunks() -> None:
    df = pl.Series("a", [False, False]).to_frame()
    df_chunked = pl.concat([df[0, :], df[1, :]], rechunk=False)
    dfi = df_chunked.__dataframe__()

    df_pd = pd.api.interchange.from_dataframe(dfi)
    result = pl.from_pandas(df_pd, nan_to_null=False)

    assert_frame_equal(result, df)


def test_roundtrip_pyarrow_boolean() -> None:
    df = pl.Series("a", [True, False], dtype=pl.Boolean).to_frame()
    dfi = df.__dataframe__()

    df_pa = pa.interchange.from_dataframe(dfi)
    result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]

    assert_frame_equal(result, df)
