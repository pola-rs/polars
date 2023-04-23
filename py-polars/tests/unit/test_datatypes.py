from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import pytest

import polars as pl
from polars import datatypes
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    DataTypeClass,
    DataTypeGroup,
    py_type_to_dtype,
)


def test_dtype_init_equivalence() -> None:
    # check "DataType.__new__" behaviour for all datatypes
    all_datatypes = {
        dtype
        for dtype in (getattr(datatypes, attr) for attr in dir(datatypes))
        if isinstance(dtype, DataTypeClass)
    }
    for dtype in all_datatypes:
        assert dtype == dtype()


def test_dtype_temporal_units() -> None:
    # check (in)equality behaviour of temporal types that take units
    for time_unit in DTYPE_TEMPORAL_UNITS:
        assert pl.Datetime == pl.Datetime(time_unit)
        assert pl.Duration == pl.Duration(time_unit)

        assert pl.Datetime(time_unit) == pl.Datetime()
        assert pl.Duration(time_unit) == pl.Duration()

    assert pl.Datetime("ms") != pl.Datetime("ns")
    assert pl.Duration("ns") != pl.Duration("us")

    # check timeunit from pytype
    for inferred_dtype, expected_dtype in (
        (py_type_to_dtype(datetime), pl.Datetime),
        (py_type_to_dtype(timedelta), pl.Duration),
    ):
        assert inferred_dtype == expected_dtype
        assert inferred_dtype.time_unit == "us"  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="Invalid time_unit"):
        pl.Datetime("?")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid time_unit"):
        pl.Duration("?")  # type: ignore[arg-type]


def test_dtype_base_type() -> None:
    assert pl.Date.base_type() is pl.Date
    assert pl.List(pl.Int32).base_type() is pl.List
    assert (
        pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)]).base_type()
        is pl.Struct
    )
    for dtype in pl.DATETIME_DTYPES:
        assert dtype.base_type() is pl.Datetime


def test_dtype_groups() -> None:
    grp = DataTypeGroup([pl.Datetime], match_base_type=False)
    assert pl.Datetime("ms", "Asia/Tokyo") not in grp

    grp = DataTypeGroup([pl.Datetime])
    assert pl.Datetime("ms", "Asia/Tokyo") in grp


def test_get_index_type() -> None:
    assert pl.get_index_type() == pl.UInt32


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
    ("dtype", "representation"),
    [
        (pl.Boolean, "Boolean"),
        (pl.Datetime, "Datetime"),
        (
            pl.Datetime(time_zone="Europe/Amsterdam"),
            "Datetime(time_unit='us', time_zone='Europe/Amsterdam')",
        ),
        (pl.List(pl.Int8), "List(Int8)"),
        (pl.List(pl.Duration(time_unit="ns")), "List(Duration(time_unit='ns'))"),
        (pl.Struct, "Struct"),
        (
            pl.Struct({"name": pl.Utf8, "ids": pl.List(pl.UInt32)}),
            "Struct([Field('name', Utf8), Field('ids', List(UInt32))])",
        ),
    ],
)
def test_repr(dtype: pl.PolarsDataType, representation: str) -> None:
    assert repr(dtype) == representation
