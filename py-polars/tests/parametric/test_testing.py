# ------------------------------------------------
# Test/validate Polars' hypothesis strategy units
# ------------------------------------------------
from __future__ import annotations

from hypothesis import given, settings
from hypothesis.strategies import sampled_from

import polars as pl
from polars.testing import column, columns, dataframes, series, strategy_dtypes

# TODO: make dtype categories flexible and available from datatypes module
TEMPORAL_DTYPES = [pl.Datetime, pl.Date, pl.Time, pl.Duration]


@given(df=dataframes(), lf=dataframes(lazy=True), srs=series())
@settings(max_examples=10)
def test_strategy_classes(df: pl.DataFrame, lf: pl.LazyFrame, srs: pl.Series) -> None:
    assert isinstance(df, pl.DataFrame)
    assert isinstance(lf, pl.LazyFrame)
    assert isinstance(srs, pl.Series)


@given(
    df1=dataframes(cols=5, size=5),
    df2=dataframes(min_cols=2, max_cols=5, min_size=3, max_size=8),
    s1=series(size=5),
    s2=series(min_size=3, max_size=8, name="col"),
)
def test_strategy_shape(
    df1: pl.DataFrame, df2: pl.DataFrame, s1: pl.Series, s2: pl.Series
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
    df=dataframes(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    lf=dataframes(excluded_dtypes=TEMPORAL_DTYPES, max_size=1, lazy=True),
    s1=series(max_size=1),
    s2=series(dtype=pl.Boolean, max_size=1),
    s3=series(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    s4=series(excluded_dtypes=TEMPORAL_DTYPES, max_size=1),
)
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
