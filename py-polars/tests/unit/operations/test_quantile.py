from statistics import quantiles

import numpy as np
import pytest
from hypothesis import given, reproduce_failure
from hypothesis.strategies import booleans, floats

import polars as pl
from polars.datatypes import FLOAT_DTYPES, INTEGER_DTYPES
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series, dtypes, series

TEMPORAL_DTYPES = [pl.Duration]


# default parameters for testing quantile
default = series(
    name="a",
    allowed_dtypes=[*FLOAT_DTYPES, *INTEGER_DTYPES],
    min_size=1,
    allow_null=True,
    allow_infinity=False,
)


@given(
    s=default,
    quantile=floats(min_value=0.0, max_value=1.0),
)
@pytest.mark.parametrize("method", ["nearest", "higher", "lower", "midpoint", "linear"])
def test_quantile_df_numeric(s: pl.Series, quantile: float, method: str):
    # test Series
    result = s.quantile(quantile, interpolation=method)

    # we must remove nulls prior to passing to numpy
    s_no_null = s.drop_nulls()
    if s_no_null.is_empty():
        assert result is None
        return

    expected = np.quantile(s_no_null.to_list(), quantile, method=method)

    # Series
    assert result == expected

    # Expr
    assert (
        s.to_frame().select(pl.col("a").quantile(quantile, interpolation=method)).item()
        == expected
    )

    # Lazy
    assert (
        s.to_frame()
        .lazy()
        .select(pl.col("a").quantile(quantile, interpolation=method))
        .collect()
        .item()
        == expected
    )
