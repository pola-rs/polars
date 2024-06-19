import pytest

import polars as pl
from polars.exceptions import ComputeError


def test_init_nonexistent_attribute() -> None:
    with pytest.raises(
        AttributeError, match="module 'polars' has no attribute 'stroopwafel'"
    ):
        pl.stroopwafel


def test_init_exceptions_deprecated() -> None:
    with pytest.deprecated_call(
        match="Accessing `ComputeError` from the top-level `polars` module is deprecated."
    ):
        exc = pl.ComputeError

    msg = "nope"
    with pytest.raises(ComputeError, match=msg):
        raise exc(msg)


def test_dtype_groups_deprecated() -> None:
    with pytest.deprecated_call(match="`INTEGER_DTYPES` is deprecated."):
        dtypes = pl.INTEGER_DTYPES

    assert pl.Int8 in dtypes
