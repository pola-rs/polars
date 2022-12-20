from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import pytest

import polars as pl
from polars import datatypes


def test_dtype_init_equivalence() -> None:
    # check "DataType.__new__" behaviour for all datatypes
    all_datatypes = {
        dtype
        for dtype in (getattr(datatypes, attr) for attr in dir(datatypes))
        if isinstance(dtype, datatypes.DataTypeClass)
    }
    for dtype in all_datatypes:
        assert dtype == dtype()


def test_dtype_temporal_units() -> None:
    # check (in)equality behaviour of temporal types that take units
    for tu in datatypes.DTYPE_TEMPORAL_UNITS:
        assert pl.Datetime == pl.Datetime(tu)
        assert pl.Duration == pl.Duration(tu)

        assert pl.Datetime(tu) == pl.Datetime()
        assert pl.Duration(tu) == pl.Duration()

    assert pl.Datetime("ms") != pl.Datetime("ns")
    assert pl.Duration("ns") != pl.Duration("us")

    # check timeunit from pytype
    for inferred_dtype, expected_dtype in (
        (datatypes.py_type_to_dtype(datetime), pl.Datetime),
        (datatypes.py_type_to_dtype(timedelta), pl.Duration),
    ):
        assert inferred_dtype == expected_dtype
        assert inferred_dtype.tu == "us"  # type: ignore[union-attr]


def test_get_idx_type() -> None:
    assert datatypes.get_idx_type() == datatypes.UInt32


def test_dtypes_picklable() -> None:
    parametric_type = pl.Datetime("ns")
    singleton_type = pl.Float64
    assert pickle.loads(pickle.dumps(parametric_type)) == parametric_type
    assert pickle.loads(pickle.dumps(singleton_type)) == singleton_type


def test_dtypes_hashable() -> None:
    # ensure that all the types can be hashed, and that their hashes
    # are sufficient to ensure distinct entries in a dictionary/set

    all_dtypes = [
        getattr(datatypes, d)
        for d in dir(datatypes)
        if isinstance(getattr(datatypes, d), datatypes.DataType)
    ]
    assert len(set(all_dtypes + all_dtypes)) == len(all_dtypes)
    assert len({pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")}) == 3
    assert len({pl.List, pl.List(pl.Int16), pl.List(pl.Int32), pl.List(pl.Int64)}) == 4


@pytest.mark.parametrize(
    ["dtype", "representation"],
    [
        (pl.Boolean, "Boolean"),
        (pl.Datetime, "Datetime"),
        (
            pl.Datetime(time_zone="Europe/Amsterdam"),
            "Datetime(tu='us', tz='Europe/Amsterdam')",
        ),
        (pl.List(pl.Int8), "List(Int8)"),
        (pl.List(pl.Duration(time_unit="ns")), "List(Duration(tu='ns'))"),
        (pl.Struct, "Struct"),
        (
            pl.Struct({"name": pl.Utf8, "ids": pl.List(pl.UInt32)}),
            "Struct([Field('name': Utf8), Field('ids': List(UInt32))])",
        ),
    ],
)
def test_repr(dtype: pl.PolarsDataType, representation: str) -> None:
    assert repr(dtype) == representation
