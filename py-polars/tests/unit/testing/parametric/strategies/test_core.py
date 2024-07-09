from datetime import datetime
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.errors import InvalidArgument

import polars as pl
from polars.testing.parametric import (
    column,
    dataframes,
    dtypes,
    lists,
    series,
)

TEMPORAL_DTYPES = {pl.Date, pl.Time, pl.Datetime, pl.Duration}


@given(s=series())
@settings(max_examples=5)
def test_series_defaults(s: pl.Series) -> None:
    assert isinstance(s, pl.Series)
    assert s.name == ""


@given(s=series(name="hello"))
@settings(max_examples=5)
def test_series_name(s: pl.Series) -> None:
    assert s.name == "hello"


@given(st.data())
def test_series_dtype(data: st.DataObject) -> None:
    dtype = data.draw(dtypes())
    s = data.draw(series(dtype=dtype))
    assert s.dtype == dtype


@given(s=series(dtype=pl.Enum, allow_null=False))
@settings(max_examples=5)
def test_series_dtype_enum(s: pl.Series) -> None:
    assert isinstance(s.dtype, pl.Enum)
    assert all(v in s.dtype.categories for v in s)


@given(s=series(dtype=pl.Boolean, min_size=5, max_size=5))
@settings(max_examples=5)
def test_series_size(s: pl.Series) -> None:
    assert s.len() == 5


@given(s=series(min_size=3, max_size=8))
@settings(max_examples=5)
def test_series_size_range(s: pl.Series) -> None:
    assert 3 <= s.len() <= 8


@given(s=series(allow_null=False))
def test_series_allow_null_false(s: pl.Series) -> None:
    assert not s.has_nulls()
    assert s.dtype != pl.Null


@given(s=series(allowed_dtypes=[pl.Null], allow_null=False))
def test_series_allow_null_allowed_dtypes(s: pl.Series) -> None:
    assert s.dtype == pl.Null


@given(s=series(allowed_dtypes=[pl.List(pl.Int8)], allow_null=False))
def test_series_allow_null_nested(s: pl.Series) -> None:
    for v in s:
        assert not v.has_nulls()


@given(df=dataframes())
@settings(max_examples=5)
def test_dataframes_defaults(df: pl.DataFrame) -> None:
    assert isinstance(df, pl.DataFrame)
    assert df.columns == [f"col{i}" for i in range(df.width)]


@given(lf=dataframes(lazy=True))
@settings(max_examples=5)
def test_dataframes_lazy(lf: pl.LazyFrame) -> None:
    assert isinstance(lf, pl.LazyFrame)


@given(df=dataframes(cols=3, min_size=5, max_size=5))
@settings(max_examples=5)
def test_dataframes_size(df: pl.DataFrame) -> None:
    assert df.height == 5
    assert df.width == 3


@given(df=dataframes(min_cols=2, max_cols=5, min_size=3, max_size=8))
@settings(max_examples=5)
def test_dataframes_size_range(df: pl.DataFrame) -> None:
    assert 3 <= df.height <= 8
    assert 2 <= df.width <= 5


@given(df=dataframes(cols=1, allow_null=True))
@settings(max_examples=5)
def test_dataframes_allow_null_global(df: pl.DataFrame) -> None:
    null_count = sum(df.null_count().row(0))
    assert 0 <= null_count <= df.height * df.width


@given(df=dataframes(cols=2, allow_null={"col0": True}))
@settings(max_examples=5)
def test_dataframes_allow_null_column(df: pl.DataFrame) -> None:
    null_count = sum(df.null_count().row(0))
    assert 0 <= null_count <= df.height * df.width


@given(
    df=dataframes(
        cols=1,
        allow_null=False,
        include_cols=[column(name="colx", allow_null=True)],
    )
)
def test_dataframes_allow_null_override(df: pl.DataFrame) -> None:
    assert not df.get_column("col0").has_nulls()
    assert 0 <= df.get_column("colx").null_count() <= df.height


@given(
    lf=dataframes(
        # generate lazyframes with at least one row
        lazy=True,
        min_size=1,
        allow_null=False,
        # test mix & match of bulk-assigned cols with custom cols
        cols=[column(n, dtype=pl.UInt8, unique=True) for n in ["a", "b"]],
        include_cols=[
            column("c", dtype=pl.Boolean),
            column("d", strategy=st.sampled_from(["x", "y", "z"])),
        ],
    )
)
def test_dataframes_columns(lf: pl.LazyFrame) -> None:
    assert lf.collect_schema() == {
        "a": pl.UInt8,
        "b": pl.UInt8,
        "c": pl.Boolean,
        "d": pl.String,
    }
    assert lf.collect_schema().names() == ["a", "b", "c", "d"]
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


@pytest.mark.hypothesis()
def test_column_invalid_probability() -> None:
    with pytest.deprecated_call(), pytest.raises(InvalidArgument):
        column("col", null_probability=2.0)


@pytest.mark.hypothesis()
def test_column_null_probability_deprecated() -> None:
    with pytest.deprecated_call():
        col = column("col", allow_null=False, null_probability=0.5)
    assert col.null_probability == 0.5
    assert col.allow_null is True  # null_probability takes precedence


@given(st.data())
def test_allow_infinities_deprecated(data: st.DataObject) -> None:
    with pytest.deprecated_call():
        strategy = series(dtype=pl.Float64, allow_infinities=False)
        s = data.draw(strategy)

    assert all(v not in (float("inf"), float("-inf")) for v in s)


@given(
    df=dataframes(
        cols=[
            column("colx", dtype=pl.Array(pl.UInt8, shape=3)),
            column("coly", dtype=pl.List(pl.Datetime("ms"))),
            column(
                name="colz",
                dtype=pl.List(pl.List(pl.String)),
                strategy=lists(
                    inner_dtype=pl.List(pl.String),
                    select_from=["aa", "bb", "cc"],
                    min_size=1,
                ),
            ),
        ],
        allow_null=False,
    ),
)
def test_dataframes_nested_strategies(df: pl.DataFrame) -> None:
    assert df.schema == {
        "colx": pl.Array(pl.UInt8, shape=3),
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


@given(
    df=dataframes(allowed_dtypes=TEMPORAL_DTYPES, max_size=1, max_cols=5),
    lf=dataframes(excluded_dtypes=TEMPORAL_DTYPES, max_size=1, max_cols=5, lazy=True),
    s1=series(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    s2=series(excluded_dtypes=TEMPORAL_DTYPES, max_size=1),
)
@settings(max_examples=50)
def test_strategy_dtypes(
    df: pl.DataFrame,
    lf: pl.LazyFrame,
    s1: pl.Series,
    s2: pl.Series,
) -> None:
    # dataframe, lazyframe
    assert all(tp.is_temporal() for tp in df.dtypes)
    assert all(not tp.is_temporal() for tp in lf.collect_schema().dtypes())

    # series
    assert s1.dtype.is_temporal()
    assert not s2.dtype.is_temporal()


@given(s=series(allow_chunks=False))
@settings(max_examples=10)
def test_series_allow_chunks(s: pl.Series) -> None:
    assert s.n_chunks() == 1


@given(df=dataframes(allow_chunks=False))
@settings(max_examples=10)
def test_dataframes_allow_chunks(df: pl.DataFrame) -> None:
    assert df.n_chunks("first") == 1
    assert df.n_chunks("all") == [1] * df.width


@given(
    df=dataframes(
        allowed_dtypes=[pl.Float32, pl.Float64],
        max_cols=4,
        allow_null=False,
        allow_infinity=False,
    ),
    s=series(dtype=pl.Float64, allow_null=False, allow_infinity=False),
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
        cols=10,
        max_size=1,
        allowed_dtypes=[pl.Int8, pl.UInt16, pl.List(pl.Int32)],
    )
)
@settings(max_examples=3)
def test_dataframes_allowed_dtypes_integer_cols(df: pl.DataFrame) -> None:
    # ensure dtype constraint works in conjunction with 'n' cols
    assert all(
        tp in (pl.Int8, pl.UInt16, pl.List(pl.Int32)) for tp in df.schema.values()
    )


@given(st.data())
@settings(max_examples=1)
def test_series_chunked_deprecated(data: st.DataObject) -> None:
    with pytest.deprecated_call():
        data.draw(series(chunked=True))
    with pytest.deprecated_call():
        data.draw(dataframes(chunked=True))
