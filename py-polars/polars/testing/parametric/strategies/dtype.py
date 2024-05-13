from __future__ import annotations

from typing import TYPE_CHECKING, Collection, Sequence

import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument

from polars.datatypes import (
    Array,
    Binary,
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    String,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn, SearchStrategy

    from polars.datatypes import DataTypeClass
    from polars.type_aliases import CategoricalOrdering, PolarsDataType, TimeUnit


# Supported data type classes which do not take any arguments
_SIMPLE_DTYPES: list[DataTypeClass] = [
    Int64,
    Int32,
    Int16,
    Int8,
    Float64,
    Float32,
    Boolean,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    String,
    Binary,
    Date,
    Time,
    Null,
]
# Supported data type classes with arguments
_COMPLEX_DTYPES: list[DataTypeClass] = [
    Datetime,
    Duration,
    Categorical,
    Decimal,
    Enum,
]
# Supported data type classes that contain other data types
_NESTED_DTYPES: list[DataTypeClass] = [
    # TODO: Enable nested types by default when various issues are solved.
    # List,
    # Array,
    Struct,
]
# Supported data type classes that do not contain other data types
_FLAT_DTYPES = _SIMPLE_DTYPES + _COMPLEX_DTYPES

_DEFAULT_ARRAY_WIDTH_LIMIT = 3
_DEFAULT_STRUCT_FIELDS_LIMIT = 3
_DEFAULT_ENUM_CATEGORIES_LIMIT = 3


def dtypes(
    *,
    allowed_dtypes: Collection[PolarsDataType] | None = None,
    excluded_dtypes: Sequence[PolarsDataType] | None = None,
    nesting_level: int = 3,
) -> SearchStrategy[DataType]:
    """
    Create a strategy for generating Polars :class:`DataType` objects.

    Parameters
    ----------
    allowed_dtypes
        Data types the strategy will pick from. If set to `None` (default),
        all supported data types are included.
    excluded_dtypes
        Data types the strategy will *not* pick from. This takes priority over
        data types specified in `allowed_dtypes`.
    nesting_level
        The complexity of nested data types. If set to 0, nested data types are
        disabled.
    """
    flat_dtypes, nested_dtypes = _parse_allowed_dtypes(allowed_dtypes)

    if nesting_level > 0 and nested_dtypes:
        if not flat_dtypes:
            return _nested_dtypes(
                inner=st.just(Null()),
                allowed_dtypes=nested_dtypes,
                excluded_dtypes=excluded_dtypes,
            )
        return st.recursive(
            base=_flat_dtypes(
                allowed_dtypes=flat_dtypes, excluded_dtypes=excluded_dtypes
            ),
            extend=lambda s: _nested_dtypes(
                s, allowed_dtypes=nested_dtypes, excluded_dtypes=excluded_dtypes
            ),
            max_leaves=nesting_level,
        )
    else:
        return _flat_dtypes(allowed_dtypes=flat_dtypes, excluded_dtypes=excluded_dtypes)


def _parse_allowed_dtypes(
    allowed_dtypes: Collection[PolarsDataType] | None = None,
) -> tuple[Sequence[PolarsDataType], Sequence[PolarsDataType]]:
    """Split allowed dtypes into flat and nested data types."""
    if allowed_dtypes is None:
        return _FLAT_DTYPES, _NESTED_DTYPES

    allowed_dtypes_flat = []
    allowed_dtypes_nested = []
    for dt in allowed_dtypes:
        if dt.is_nested():
            allowed_dtypes_nested.append(dt)
        else:
            allowed_dtypes_flat.append(dt)

    return allowed_dtypes_flat, allowed_dtypes_nested


@st.composite
def _flat_dtypes(
    draw: DrawFn,
    allowed_dtypes: Sequence[PolarsDataType] | None = None,
    excluded_dtypes: Sequence[PolarsDataType] | None = None,
) -> DataType:
    """Create a strategy for generating non-nested Polars :class:`DataType` objects."""
    if allowed_dtypes is None:
        allowed_dtypes = _FLAT_DTYPES
    if excluded_dtypes is None:
        excluded_dtypes = []

    dtype = draw(st.sampled_from(allowed_dtypes))
    return draw(
        _instantiate_flat_dtype(dtype).filter(lambda x: x not in excluded_dtypes)
    )


@st.composite
def _instantiate_flat_dtype(draw: DrawFn, dtype: PolarsDataType) -> DataType:
    """Take a flat data type and instantiate it."""
    if isinstance(dtype, DataType):
        return dtype
    elif dtype in _SIMPLE_DTYPES:
        return dtype()
    elif dtype == Datetime:
        # TODO: Add time zones
        time_unit = draw(_time_units())
        return Datetime(time_unit)
    elif dtype == Duration:
        time_unit = draw(_time_units())
        return Duration(time_unit)
    elif dtype == Categorical:
        ordering = draw(_categorical_orderings())
        return Categorical(ordering)
    elif dtype == Enum:
        n_categories = draw(
            st.integers(min_value=1, max_value=_DEFAULT_ENUM_CATEGORIES_LIMIT)
        )
        categories = [f"c{i}" for i in range(n_categories)]
        return Enum(categories)
    elif dtype == Decimal:
        precision = draw(st.integers(min_value=1, max_value=38) | st.none())
        scale = draw(st.integers(min_value=0, max_value=precision or 38))
        return Decimal(precision, scale)
    else:
        msg = f"unsupported data type: {dtype}"
        raise InvalidArgument(msg)


@st.composite
def _nested_dtypes(
    draw: DrawFn,
    inner: SearchStrategy[DataType],
    allowed_dtypes: Sequence[PolarsDataType] | None = None,
    excluded_dtypes: Sequence[PolarsDataType] | None = None,
) -> DataType:
    """Create a strategy for generating nested Polars :class:`DataType` objects."""
    if allowed_dtypes is None:
        allowed_dtypes = _NESTED_DTYPES
    if excluded_dtypes is None:
        excluded_dtypes = []

    dtype = draw(st.sampled_from(allowed_dtypes))
    return draw(
        _instantiate_nested_dtype(dtype, inner).filter(
            lambda x: x not in excluded_dtypes
        )
    )


@st.composite
def _instantiate_nested_dtype(
    draw: DrawFn,
    dtype: PolarsDataType,
    inner: SearchStrategy[DataType],
) -> DataType:
    """Take a nested data type and instantiate it."""

    def instantiate_inner(dtype: PolarsDataType) -> DataType:
        inner_dtype = getattr(dtype, "inner", None)
        if inner_dtype is None:
            return draw(inner)
        elif inner_dtype.is_nested():
            return draw(_instantiate_nested_dtype(inner_dtype, inner))
        else:
            return draw(_instantiate_flat_dtype(inner_dtype))

    if dtype == List:
        inner_dtype = instantiate_inner(dtype)
        return List(inner_dtype)
    elif dtype == Array:
        inner_dtype = instantiate_inner(dtype)
        width = getattr(
            dtype,
            "width",
            draw(st.integers(min_value=1, max_value=_DEFAULT_ARRAY_WIDTH_LIMIT)),
        )
        return Array(inner_dtype, width)
    elif dtype == Struct:
        # TODO: Recursively instantiate struct field dtypes
        if isinstance(dtype, DataType):
            return dtype
        n_fields = draw(
            st.integers(min_value=1, max_value=_DEFAULT_STRUCT_FIELDS_LIMIT)
        )
        return Struct({f"f{i}": draw(inner) for i in range(n_fields)})
    else:
        msg = f"unsupported data type: {dtype}"
        raise InvalidArgument(msg)


def _time_units() -> SearchStrategy[TimeUnit]:
    """Create a strategy for generating valid units of time."""
    return st.sampled_from(["us", "ns", "ms"])


def _categorical_orderings() -> SearchStrategy[CategoricalOrdering]:
    """Create a strategy for generating valid ordering types for categorical data."""
    return st.sampled_from(["physical", "lexical"])


@st.composite
def _instantiate_dtype(
    draw: DrawFn,
    dtype: PolarsDataType,
    *,
    allowed_dtypes: Collection[PolarsDataType] | None = None,
    excluded_dtypes: Sequence[PolarsDataType] | None = None,
    nesting_level: int = 3,
) -> DataType:
    """Take a data type and instantiate it."""
    if not dtype.is_nested():
        if allowed_dtypes is None:
            allowed_dtypes = [dtype]
        else:
            allowed_dtypes = [dt for dt in allowed_dtypes if dt == dtype]
        return draw(
            _flat_dtypes(allowed_dtypes=allowed_dtypes, excluded_dtypes=excluded_dtypes)
        )

    def draw_inner(dtype: PolarsDataType) -> DataType:
        if isinstance(dtype, DataType):
            return draw(
                _instantiate_dtype(
                    dtype.inner,  # type: ignore[attr-defined]
                    allowed_dtypes=allowed_dtypes,
                    excluded_dtypes=excluded_dtypes,
                    nesting_level=nesting_level - 1,
                )
            )
        else:
            return draw(
                dtypes(
                    allowed_dtypes=allowed_dtypes,
                    excluded_dtypes=excluded_dtypes,
                    nesting_level=nesting_level - 1,
                )
            )

    if dtype == List:
        inner = draw_inner(dtype)
        return List(inner)
    elif dtype == Array:
        inner = draw_inner(dtype)
        width = getattr(
            dtype,
            "width",
            draw(st.integers(min_value=1, max_value=_DEFAULT_ARRAY_WIDTH_LIMIT)),
        )
        return Array(inner, width)
    elif dtype == Struct:
        if isinstance(dtype, DataType):
            return dtype
        n_fields = draw(
            st.integers(min_value=1, max_value=_DEFAULT_STRUCT_FIELDS_LIMIT)
        )
        inner_strategy = dtypes(
            allowed_dtypes=allowed_dtypes,
            excluded_dtypes=excluded_dtypes,
            nesting_level=nesting_level - 1,
        )
        return Struct({f"f{i}": draw(inner_strategy) for i in range(n_fields)})
    else:
        msg = f"unsupported data type: {dtype}"
        raise InvalidArgument(msg)
