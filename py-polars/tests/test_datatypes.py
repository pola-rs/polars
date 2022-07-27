from __future__ import annotations

import inspect

import polars as pl
import polars.datatypes as datatypes


def test_dtype_init_equivalence() -> None:
    # check "DataType.__new__" behaviour for all datatypes
    all_datatypes = {
        dtype
        for dtype in (getattr(datatypes, attr) for attr in dir(datatypes))
        if inspect.isclass(dtype) and issubclass(dtype, datatypes.DataType)
    }
    for dtype in all_datatypes:
        assert dtype == dtype()


def test_dtype_temporal_units() -> None:
    # check (in)equality behaviour of temporal types that take units
    for tu in datatypes.DTYPE_TEMPORAL_UNITS:
        assert pl.Datetime == pl.Datetime(tu)
        assert pl.Duration == pl.Duration(tu)

        assert pl.Datetime(tu) == pl.Datetime()  # type: ignore[operator]
        assert pl.Duration(tu) == pl.Duration()  # type: ignore[operator]

    assert pl.Datetime("ms") != pl.Datetime("ns")
    assert pl.Duration("ns") != pl.Duration("us")


def test_get_idx_type() -> None:
    assert datatypes.get_idx_type() == datatypes.UInt32
