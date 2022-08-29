from __future__ import annotations

import inspect
import pickle

import polars as pl
from polars import datatypes


def test_dtype_init_equivalence() -> None:
    # check "DataType.__new__" behaviour for all datatypes
    all_datatypes = {
        dtype
        for dtype in (getattr(datatypes, attr) for attr in dir(datatypes))
        if inspect.isclass(dtype) and issubclass(dtype, datatypes.DataType)
    }
    for dtype in all_datatypes:
        assert dtype == dtype()  # type: ignore[comparison-overlap]


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


def test_dtypes_picklable() -> None:
    parametric_type = pl.Datetime("ns")
    singleton_type = pl.Float64
    assert pickle.loads(pickle.dumps(parametric_type)) == parametric_type
    assert pickle.loads(pickle.dumps(singleton_type)) == singleton_type
