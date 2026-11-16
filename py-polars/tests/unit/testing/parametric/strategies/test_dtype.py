from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis.strategies as st
from hypothesis import given

import polars as pl
from polars.testing.parametric.strategies.dtype import dtypes

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@given(dtype=dtypes())
def test_dtypes(dtype: pl.DataType) -> None:
    assert isinstance(dtype, pl.DataType)


@given(dtype=dtypes(nesting_level=0))
def test_dtypes_nesting_level(dtype: pl.DataType) -> None:
    assert not dtype.is_nested()


@given(dtype=dtypes(allowed_dtypes=[pl.Datetime], allow_time_zones=False))
def test_dtypes_allow_time_zones(dtype: pl.DataType) -> None:
    assert getattr(dtype, "time_zone", None) is None


@given(st.data())
def test_dtypes_allowed(data: st.DataObject) -> None:
    allowed_dtype = data.draw(dtypes())
    result = data.draw(dtypes(allowed_dtypes=[allowed_dtype]))
    assert result == allowed_dtype


@given(st.data())
def test_dtypes_excluded(data: st.DataObject) -> None:
    excluded_dtype = data.draw(dtypes())
    result = data.draw(dtypes(excluded_dtypes=[excluded_dtype]))
    assert result != excluded_dtype


@given(dtype=dtypes(allowed_dtypes=[pl.Duration], excluded_dtypes=[pl.Duration("ms")]))
def test_dtypes_allowed_excluded_instance(dtype: pl.DataType) -> None:
    assert isinstance(dtype, pl.Duration)
    assert dtype.time_unit != "ms"


@given(
    dtype=dtypes(
        allowed_dtypes=[pl.Duration("ns"), pl.Date], excluded_dtypes=[pl.Duration]
    )
)
def test_dtypes_allowed_excluded_priority(dtype: pl.DataType) -> None:
    assert dtype == pl.Date


@given(dtype=dtypes(allowed_dtypes=[pl.Int8(), pl.Duration("ms")]))
def test_dtypes_allowed_instantiated(dtype: pl.DataType) -> None:
    assert dtype in (pl.Int8(), pl.Duration("ms"))


@given(dtype=dtypes(allowed_dtypes=[pl.List(pl.List), pl.Int64]))
def test_dtypes_allowed_uninstantiated_nested(dtype: pl.DataType) -> None:
    assert dtype in (pl.List, pl.Int64)


def _union_kind(dt: PolarsDataType) -> object:
    if dt.is_integer():
        return "int"
    if dt.is_float():
        return "float"
    if dt in (pl.String, pl.Enum):
        return "str"
    return dt.base_type()


@given(
    dtype=dtypes(
        allowed_dtypes=[pl.Union, pl.Int8, pl.Float32, pl.String, pl.Enum, pl.List],
        nesting_level=2,
    )
)
def test_dtypes_union(dtype: pl.DataType) -> None:
    if isinstance(dtype, pl.Union):
        # A Python value must pick one variant, so the variants have distinct kinds.
        kinds = [_union_kind(f.dtype) for f in dtype.fields]
        assert 1 <= len(kinds) <= 3
        assert len(set(kinds)) == len(kinds)
