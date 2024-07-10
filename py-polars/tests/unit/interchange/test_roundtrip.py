from __future__ import annotations

import sys
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.interchange
import pytest
from hypothesis import given

import polars as pl
from polars._utils.various import parse_version
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

protocol_dtypes: list[PolarsDataType] = [
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
    pl.String,
    pl.Datetime,
    # TODO: Enable lexically ordered categoricals
    pl.Categorical("physical"),
    # TODO: Add Enum
    # pl.Enum,
]


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        allow_null=False,  # Bug: https://github.com/pola-rs/polars/issues/16190
    )
)
def test_to_dataframe_pyarrow_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__()
    df_pa = pa.interchange.from_dataframe(dfi)

    with pl.StringCache():
        result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.String,  # Polars String type does not match protocol spec
            pl.Categorical,
        ],
        allow_chunks=False,
    )
)
def test_to_dataframe_pyarrow_zero_copy_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__(allow_copy=False)
    df_pa = pa.interchange.from_dataframe(dfi, allow_copy=False)

    result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]
    assert_frame_equal(result, df, categorical_as_str=True)


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="The correct `from_dataframe` implementation for pandas is not available before Python 3.9",
)
@pytest.mark.filterwarnings(
    "ignore:.*PEP3118 format string that does not match its itemsize:RuntimeWarning"
)
@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        allow_null=False,  # Bug: https://github.com/pola-rs/polars/issues/16190
    )
)
def test_to_dataframe_pandas_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__()
    df_pd = pd.api.interchange.from_dataframe(dfi)
    result = pl.from_pandas(df_pd, nan_to_null=False)
    assert_frame_equal(result, df, categorical_as_str=True)


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="The correct `from_dataframe` implementation for pandas is not available before Python 3.9",
)
@pytest.mark.filterwarnings(
    "ignore:.*PEP3118 format string that does not match its itemsize:RuntimeWarning"
)
@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.String,  # Polars String type does not match protocol spec
            pl.Categorical,
        ],
        allow_chunks=False,
        allow_null=False,  # Bug: https://github.com/pola-rs/polars/issues/16190
    )
)
def test_to_dataframe_pandas_zero_copy_parametric(df: pl.DataFrame) -> None:
    dfi = df.__dataframe__(allow_copy=False)
    df_pd = pd.api.interchange.from_dataframe(dfi, allow_copy=False)
    result = pl.from_pandas(df_pd, nan_to_null=False)
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.Categorical,  # Categoricals read back as Enum types
        ],
    )
)
def test_from_dataframe_pyarrow_parametric(df: pl.DataFrame) -> None:
    df_pa = df.to_arrow()
    result = pl.from_dataframe(df_pa)
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.String,  # Polars String type does not match protocol spec
            pl.Categorical,  # Polars copies the categories to construct a mapping
            pl.Boolean,  # pyarrow exports boolean buffers as byte-packed: https://github.com/apache/arrow/issues/37991
        ],
        allow_chunks=False,
    )
)
def test_from_dataframe_pyarrow_zero_copy_parametric(df: pl.DataFrame) -> None:
    df_pa = df.to_arrow()
    result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.Categorical,  # Categoricals come back as Enums
            pl.Float32,  # NaN values come back as nulls
            pl.Float64,  # NaN values come back as nulls
        ],
    )
)
@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Older versions of pandas do not implement the required conversions",
)
def test_from_dataframe_pandas_parametric(df: pl.DataFrame) -> None:
    df_pd = df.to_pandas(use_pyarrow_extension_array=True)
    result = pl.from_dataframe(df_pd)
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.String,  # Polars String type does not match protocol spec
            pl.Categorical,  # Categoricals come back as Enums
            pl.Float32,  # NaN values come back as nulls
            pl.Float64,  # NaN values come back as nulls
            pl.Boolean,  # pandas exports boolean buffers as byte-packed
        ],
        # Empty dataframes cause an error due to a bug in pandas.
        # https://github.com/pandas-dev/pandas/issues/56700
        min_size=1,
        allow_chunks=False,
    )
)
@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Older versions of pandas do not implement the required conversions",
)
def test_from_dataframe_pandas_zero_copy_parametric(df: pl.DataFrame) -> None:
    df_pd = df.to_pandas(use_pyarrow_extension_array=True)
    result = pl.from_dataframe(df_pd, allow_copy=False)
    assert_frame_equal(result, df)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.Categorical,  # Categoricals come back as Enums
            pl.Float32,  # NaN values come back as nulls
            pl.Float64,  # NaN values come back as nulls
        ],
        # Empty string columns cause an error due to a bug in pandas.
        # https://github.com/pandas-dev/pandas/issues/56703
        min_size=1,
        allow_null=False,  # Bug: https://github.com/pola-rs/polars/issues/16190
    )
)
@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Older versions of pandas do not implement the required conversions",
)
def test_from_dataframe_pandas_native_parametric(df: pl.DataFrame) -> None:
    df_pd = df.to_pandas()
    result = pl.from_dataframe(df_pd)
    assert_frame_equal(result, df, categorical_as_str=True)


@given(
    dataframes(
        allowed_dtypes=protocol_dtypes,
        excluded_dtypes=[
            pl.String,  # Polars String type does not match protocol spec
            pl.Categorical,  # Categoricals come back as Enums
            pl.Float32,  # NaN values come back as nulls
            pl.Float64,  # NaN values come back as nulls
            pl.Boolean,  # pandas exports boolean buffers as byte-packed
        ],
        # Empty dataframes cause an error due to a bug in pandas.
        # https://github.com/pandas-dev/pandas/issues/56700
        min_size=1,
        allow_chunks=False,
        allow_null=False,  # Bug: https://github.com/pola-rs/polars/issues/16190
    )
)
@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Older versions of pandas do not implement the required conversions",
)
def test_from_dataframe_pandas_native_zero_copy_parametric(df: pl.DataFrame) -> None:
    df_pd = df.to_pandas()
    result = pl.from_dataframe(df_pd, allow_copy=False)
    assert_frame_equal(result, df)


def test_to_dataframe_pandas_boolean_subchunks() -> None:
    df = pl.Series("a", [False, False]).to_frame()
    df_chunked = pl.concat([df[0, :], df[1, :]], rechunk=False)
    dfi = df_chunked.__dataframe__()

    df_pd = pd.api.interchange.from_dataframe(dfi)
    result = pl.from_pandas(df_pd, nan_to_null=False)

    assert_frame_equal(result, df)


def test_to_dataframe_pyarrow_boolean() -> None:
    df = pl.Series("a", [True, False], dtype=pl.Boolean).to_frame()
    dfi = df.__dataframe__()

    df_pa = pa.interchange.from_dataframe(dfi)
    result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]

    assert_frame_equal(result, df)


def test_to_dataframe_pyarrow_boolean_midbyte_slice() -> None:
    s = pl.Series("a", [False] * 9)[3:]
    df = s.to_frame()
    dfi = df.__dataframe__()

    df_pa = pa.interchange.from_dataframe(dfi)
    result: pl.DataFrame = pl.from_arrow(df_pa)  # type: ignore[assignment]

    assert_frame_equal(result, df)


@pytest.mark.skipif(
    parse_version(pd.__version__) < (2, 2),
    reason="Pandas versions < 2.2 do not implement the required conversions",
)
def test_from_dataframe_pandas_timestamp_ns() -> None:
    df = pl.Series("a", [datetime(2000, 1, 1)], dtype=pl.Datetime("ns")).to_frame()
    df_pd = df.to_pandas(use_pyarrow_extension_array=True)
    result = pl.from_dataframe(df_pd)
    assert_frame_equal(result, df)
