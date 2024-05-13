# ------------------------------------------------
# Test/validate Polars' hypothesis strategy units
# ------------------------------------------------
from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.errors import InvalidArgument, NonInteractiveExampleWarning

import polars as pl
from polars.testing.parametric import column, dataframes, lists, series

TEMPORAL_DTYPES = {pl.Date, pl.Time, pl.Datetime, pl.Duration}


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


@given(
    lf=dataframes(
        # generate lazyframes with at least one row
        lazy=True,
        min_size=1,
        # test mix & match of bulk-assigned cols with custom cols
        cols=[column(n, dtype=pl.UInt8, unique=True) for n in ["a", "b"]],
        include_cols=[
            column("c", dtype=pl.Boolean),
            column("d", strategy=st.sampled_from(["x", "y", "z"])),
        ],
    )
)
def test_strategy_frame_columns(lf: pl.LazyFrame) -> None:
    assert lf.schema == {"a": pl.UInt8, "b": pl.UInt8, "c": pl.Boolean, "d": pl.String}
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
    s1=series(dtype=pl.Boolean, max_size=1),
    s2=series(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    s3=series(excluded_dtypes=TEMPORAL_DTYPES, max_size=1),
)
@settings(max_examples=50)
def test_strategy_dtypes(
    df: pl.DataFrame,
    lf: pl.LazyFrame,
    s1: pl.Series,
    s2: pl.Series,
    s3: pl.Series,
) -> None:
    # dataframe, lazyframe
    assert all(tp.is_temporal() for tp in df.dtypes)
    assert all(not tp.is_temporal() for tp in lf.dtypes)

    # series
    assert s1.dtype == pl.Boolean
    assert s2.dtype.is_temporal()
    assert not s3.dtype.is_temporal()


@given(s=series())
def test_series_null_probability_default(s: pl.Series) -> None:
    assert s.null_count() == 0


@given(s=series(null_probability=0.1))
def test_series_null_probability(s: pl.Series) -> None:
    assert 0 <= s.null_count() <= s.len()


@given(df=dataframes(cols=1, null_probability=0.3))
def test_dataframes_null_probability_global(df: pl.DataFrame) -> None:
    null_count = sum(df.null_count().row(0))
    assert 0 <= null_count <= df.height * df.width


@given(df=dataframes(cols=2, null_probability={"col0": 0.7}))
def test_dataframes_null_probability_column(df: pl.DataFrame) -> None:
    null_count = sum(df.null_count().row(0))
    assert 0 <= null_count <= df.height * df.width


@given(
    df=dataframes(
        cols=1,
        null_probability=1.0,
        include_cols=[column(name="colx", null_probability=0.2)],
    )
)
def test_dataframes_null_probability_override(df: pl.DataFrame) -> None:
    assert df.get_column("col0").null_count() == df.height
    assert 0 <= df.get_column("col0").null_count() <= df.height


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
        allowed_dtypes=[pl.Float32, pl.Float64],
        allow_infinities=False,
        max_cols=4,
    ),
    s=series(dtype=pl.Float64, allow_infinities=False),
)
def test_infinities(
    df: pl.DataFrame,
    s: pl.Series,
) -> None:
    from math import isfinite, isnan

    def finite_float(value: Any) -> bool:
        return isfinite(value) or isnan(value)

    assert all(finite_float(val) for val in s.to_list())
    for col in df.columns:
        assert all(finite_float(val) for val in df[col].to_list())


@given(
    df=dataframes(
        cols=[
            column("colx", dtype=pl.Array(pl.UInt8, width=3)),
            column("coly", dtype=pl.List(pl.Datetime("ms"))),
            column(
                name="colz",
                dtype=pl.List(pl.List(pl.String)),
                strategy=lists(
                    inner_dtype=pl.List(pl.String),
                    select_from=["aa", "bb", "cc"],
                    min_len=1,
                ),
            ),
        ]
    ),
)
def test_sequence_strategies(df: pl.DataFrame) -> None:
    assert df.schema == {
        "colx": pl.Array(pl.UInt8, width=3),
        "coly": pl.List(pl.Datetime("ms")),
        "colz": pl.List(pl.List(pl.String)),
    }
    uint8_max = (2**8) - 1

    for colx, coly, colz in df.iter_rows():
        assert len(colx) == 3
        assert all(i <= uint8_max for i in colx)
        assert all(isinstance(d, datetime) for d in coly)
        for inner_list in colz:
            assert all(s in ("aa", "bb", "cc") for s in inner_list)


@pytest.mark.hypothesis()
@pytest.mark.parametrize("invalid_probability", [-1.0, +2.0])
def test_invalid_argument_null_probability(invalid_probability: float) -> None:
    with pytest.raises(InvalidArgument, match="between 0.0 and 1.0"):
        column("colx", dtype=pl.Boolean, null_probability=invalid_probability)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        with pytest.raises(InvalidArgument, match="between 0.0 and 1.0"):
            series(name="colx", null_probability=invalid_probability).example()
        with pytest.raises(InvalidArgument, match="between 0.0 and 1.0"):
            dataframes(
                cols=column(None),
                null_probability=invalid_probability,
            ).example()


@pytest.mark.hypothesis()
def test_column_invalid_probability() -> None:
    with pytest.raises(InvalidArgument):
        column("col", null_probability=2.0)
