# ------------------------------------------------
# Test/validate Polars' hypothesis strategy units
# ------------------------------------------------
from __future__ import annotations

import warnings
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis.errors import InvalidArgument, NonInteractiveExampleWarning
from hypothesis.strategies import sampled_from

import polars as pl
from polars.testing.parametric import (
    column,
    columns,
    dataframes,
    series,
    strategy_dtypes,
)

# TODO: make dtype categories flexible and available from datatypes module
TEMPORAL_DTYPES = [pl.Datetime, pl.Date, pl.Time, pl.Duration]


@given(df=dataframes(), lf=dataframes(lazy=True), srs=series())
@settings(max_examples=5)
def test_strategy_classes(df: pl.DataFrame, lf: pl.LazyFrame, srs: pl.Series) -> None:
    assert isinstance(df, pl.DataFrame)
    assert isinstance(lf, pl.LazyFrame)
    assert isinstance(srs, pl.Series)


@given(
    s1=series(dtype=pl.Boolean, size=5),
    s2=series(dtype=pl.Boolean, min_size=3, max_size=8, name="col"),
    df1=dataframes(allowed_dtypes=[pl.Boolean], cols=5, size=5),
    df2=dataframes(
        allowed_dtypes=[pl.Boolean], min_cols=2, max_cols=5, min_size=3, max_size=8
    ),
)
@settings(max_examples=50)
def test_strategy_shape(
    s1: pl.Series, s2: pl.Series, df1: pl.DataFrame, df2: pl.DataFrame
) -> None:
    assert df1.shape == (5, 5)
    assert df1.columns == ["col0", "col1", "col2", "col3", "col4"]

    assert 2 <= len(df2.columns) <= 5
    assert 3 <= len(df2) <= 8

    assert s1.len() == 5
    assert 3 <= s2.len() <= 8
    assert s1.name == ""
    assert s2.name == "col"

    from polars.testing._parametric import MAX_COLS

    assert 0 <= len(columns(None)) <= MAX_COLS


@given(
    lf=dataframes(
        # generate lazyframes with at least one row
        lazy=True,
        min_size=1,
        # test mix & match of bulk-assigned cols with custom cols
        cols=columns(["a", "b"], dtype=pl.UInt8, unique=True),
        include_cols=[
            column("c", dtype=pl.Boolean),
            column("d", strategy=sampled_from(["x", "y", "z"])),
        ],
    )
)
def test_strategy_frame_columns(lf: pl.LazyFrame) -> None:
    assert lf.schema == {"a": pl.UInt8, "b": pl.UInt8, "c": pl.Boolean, "d": pl.Utf8}
    assert lf.columns == ["a", "b", "c", "d"]
    df = lf.collect()

    # confirm uint cols bounds
    uint8_max = (2**8) - 1
    assert df["a"].min() >= 0  # type: ignore[operator]
    assert df["b"].min() >= 0  # type: ignore[operator]
    assert df["a"].max() <= uint8_max  # type: ignore[operator]
    assert df["b"].max() <= uint8_max  # type: ignore[operator]

    # confirm uint cols uniqueness
    assert df["a"].is_unique().all()
    assert df["b"].is_unique().all()

    # boolean col
    assert all(isinstance(v, bool) for v in df["c"].to_list())

    # string col, entries selected from custom values
    xyz = {"x", "y", "z"}
    assert all(v in xyz for v in df["d"].to_list())


@given(
    df=dataframes(allowed_dtypes=TEMPORAL_DTYPES, max_size=1, max_cols=5),
    lf=dataframes(excluded_dtypes=TEMPORAL_DTYPES, max_size=1, max_cols=5, lazy=True),
    s1=series(max_size=1),
    s2=series(dtype=pl.Boolean, max_size=1),
    s3=series(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    s4=series(excluded_dtypes=TEMPORAL_DTYPES, max_size=1),
)
@settings(max_examples=50)
def test_strategy_dtypes(
    df: pl.DataFrame,
    lf: pl.LazyFrame,
    s1: pl.Series,
    s2: pl.Series,
    s3: pl.Series,
    s4: pl.Series,
) -> None:
    # dataframe, lazyframe
    assert all(tp in TEMPORAL_DTYPES for tp in df.dtypes)
    assert all(tp not in TEMPORAL_DTYPES for tp in lf.dtypes)

    # series
    assert s1.dtype in strategy_dtypes
    assert s2.dtype == pl.Boolean
    assert s3.dtype in TEMPORAL_DTYPES
    assert s4.dtype not in TEMPORAL_DTYPES


@given(
    # set global, per-column, and overridden null-probabilities
    s=series(size=50, null_probability=0.10),
    df1=dataframes(cols=1, size=50, null_probability=0.30),
    df2=dataframes(cols=2, size=50, null_probability={"col0": 0.70}),
    df3=dataframes(
        cols=1,
        size=50,
        null_probability=1.0,
        include_cols=[column(name="colx", null_probability=0.20)],
    ),
)
@settings(max_examples=50)
def test_strategy_null_probability(
    s: pl.Series,
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    df3: pl.DataFrame,
) -> None:
    for obj in (s, df1, df2, df3):
        assert len(obj) == 50  # type: ignore[arg-type]

    assert s.null_count() < df1.null_count().fold(sum).sum()
    assert df1.null_count().fold(sum).sum() < df2.null_count().fold(sum).sum()
    assert df2.null_count().fold(sum).sum() < df3.null_count().fold(sum).sum()

    nulls_col0, nulls_col1 = df2.null_count().rows()[0]
    assert nulls_col0 > nulls_col1
    assert nulls_col0 < 50

    nulls_col0, nulls_colx = df3.null_count().rows()[0]
    assert nulls_col0 > nulls_colx
    assert nulls_col0 == 50


@given(
    df1=dataframes(chunked=False, min_size=1),
    df2=dataframes(chunked=True, min_size=1),
    s1=series(chunked=False, min_size=1),
    s2=series(chunked=True, min_size=1),
)
@settings(max_examples=10)
def test_chunking(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    s1: pl.Series,
    s2: pl.Series,
) -> None:
    assert df1.n_chunks() == 1
    if len(df2) > 1:
        assert df2.n_chunks("all") == [2] * len(df2.columns)

    assert s1.n_chunks() == 1
    if len(s2) > 1:
        assert s2.n_chunks() > 1


@given(
    df=dataframes(
        allowed_dtypes=[pl.Float32, pl.Float64], max_cols=4, allow_infinities=False
    ),
    s=series(dtype=pl.Float64, allow_infinities=False),
)
def test_infinities(
    df: pl.DataFrame,
    s: pl.Series,
) -> None:
    from math import isfinite

    def finite_float(value: Any) -> bool:
        if isinstance(value, float):
            return isfinite(value)
        return False

    assert all(finite_float(val) for val in s.to_list())
    for col in df.columns:
        assert all(finite_float(val) for val in df[col].to_list())


def test_invalid_arguments() -> None:
    for invalid_probability in (-1.0, +2.0):
        with pytest.raises(InvalidArgument, match="between 0.0 and 1.0"):
            column("colx", dtype=pl.Boolean, null_probability=invalid_probability)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
            with pytest.raises(InvalidArgument, match="between 0.0 and 1.0"):
                series(name="colx", null_probability=invalid_probability).example()
            with pytest.raises(InvalidArgument, match="between 0.0 and 1.0"):
                dataframes(
                    cols=column(None),  # type: ignore[arg-type]
                    null_probability=invalid_probability,
                ).example()

    for unsupported_type in (pl.List, pl.Struct):
        # TODO: add support for compound types
        with pytest.raises(InvalidArgument, match=f"for {unsupported_type} type"):
            column("colx", dtype=unsupported_type)

    with pytest.raises(InvalidArgument, match="not a valid polars datatype"):
        columns(["colx", "coly"], dtype=pl.DataFrame)  # type: ignore[arg-type]

    with pytest.raises(InvalidArgument, match=r"\d dtypes for \d names"):
        columns(["colx", "coly"], dtype=[pl.Date, pl.Date, pl.Datetime])

    with pytest.raises(InvalidArgument, match="Unable to determine dtype"):
        column("colx", strategy=sampled_from([None]))
