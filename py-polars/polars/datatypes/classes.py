from __future__ import annotations

import contextlib
from inspect import isclass
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import polars.datatypes as dt

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import dtype_str_repr as _dtype_str_repr


if TYPE_CHECKING:
    import sys

    from polars.type_aliases import PolarsDataType, PythonDataType, SchemaDict, TimeUnit

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class DataType:
    """Base class for all Polars data types."""

    def __eq__(self, other: Any) -> bool:
        # allow comparing object instances to class
        if isclass(other) and issubclass(other, DataType):
            return self.__class__.__name__ == other.__name__
        elif isinstance(other, DataType):
            return self.__class__.__name__ == other.__class__.__name__
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __reduce__(self) -> Any:
        return (_custom_reconstruct, (type(self), object, None), self.__dict__)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def _string_repr(self) -> str:
        return _dtype_str_repr(self)

    @classmethod
    def base_type(cls) -> type[Self]:
        """
        Return this DataType's fundamental/root type class.

        Examples
        --------
        >>> pl.Datetime("ns").base_type()
        <class 'polars.datatypes.classes.Datetime'>
        >>> pl.List(pl.Int32).base_type()
        <class 'polars.datatypes.classes.List'>
        >>> pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)]).base_type()
        <class 'polars.datatypes.classes.Struct'>
        """
        return cls


def _custom_reconstruct(
    cls: type[Any], base: type[Any], state: Any
) -> PolarsDataType | type:
    """Helper function for unpickling DataType objects."""
    if state:
        obj = base.__new__(cls, state)
        if base.__init__ != object.__init__:
            base.__init__(obj, state)
    else:
        obj = object.__new__(cls)
    return obj


class NumericType(DataType):
    """Base class for numeric data types."""


class IntegralType(NumericType):
    """Base class for integral data types."""


class FractionalType(NumericType):
    """Base class for fractional data types."""


class FloatType(FractionalType):
    """Base class for float data types."""


class TemporalType(DataType):
    """Base class for temporal data types."""


class NestedType(DataType):
    """Base class for nested data types."""


class Int8(IntegralType):
    """8-bit signed integer type."""


class Int16(IntegralType):
    """16-bit signed integer type."""


class Int32(IntegralType):
    """32-bit signed integer type."""


class Int64(IntegralType):
    """64-bit signed integer type."""


class UInt8(IntegralType):
    """8-bit unsigned integer type."""


class UInt16(IntegralType):
    """16-bit unsigned integer type."""


class UInt32(IntegralType):
    """32-bit unsigned integer type."""


class UInt64(IntegralType):
    """64-bit unsigned integer type."""


class Float32(FloatType):
    """32-bit floating point type."""


class Float64(FloatType):
    """64-bit floating point type."""


class Decimal(FractionalType):
    """
    Decimal 128-bit type with an optional precision and non-negative scale.

    NOTE: this is an experimental work-in-progress feature and may not work as expected.
    """

    def __init__(self, precision: int | None = None, scale: int = 0):
        """
        Decimal type.

        Parameters
        ----------
        precision
            ...
        scale
            ...

        """
        self.precision = precision
        self.scale = scale

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(precision={self.precision}, scale={self.scale})"
        )

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if isclass(other) and issubclass(other, Decimal):
            return True
        elif isinstance(other, Decimal):
            return self.precision == other.precision and self.scale == other.scale
        else:
            return False

    def __hash__(self) -> int:
        return hash((Decimal, self.precision, self.scale))


class Boolean(DataType):
    """Boolean type."""


class Utf8(DataType):
    """UTF-8 encoded string type."""


class Binary(DataType):
    """Binary type."""


class Date(TemporalType):
    """Calendar date type."""


class Time(TemporalType):
    """Time of day type."""


class Datetime(TemporalType):
    """Calendar date and time type."""

    def __init__(self, time_unit: TimeUnit = "us", time_zone: str | None = None):
        """
        Calendar date and time type.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms'}
            Unit of time.
        time_zone
            Time zone string as defined in zoneinfo (run
            ``import zoneinfo; zoneinfo.available_timezones()`` for a full list).

        """
        if time_unit not in ("ms", "us", "ns"):
            raise ValueError(
                f"Invalid time_unit; expected one of {{'ns','us','ms'}}, got {time_unit!r}"
            )

        self.time_unit = time_unit
        self.time_zone = time_zone

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if isclass(other) and issubclass(other, Datetime):
            return True
        elif isinstance(other, Datetime):
            return (
                self.time_unit == other.time_unit and self.time_zone == other.time_zone
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((Datetime, self.time_unit))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(time_unit={self.time_unit!r}, time_zone={self.time_zone!r})"
        )


class Duration(TemporalType):
    """Time duration/delta type."""

    def __init__(self, time_unit: TimeUnit = "us"):
        """
        Time duration/delta type.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms'}
            Unit of time.

        """
        if time_unit not in ("ms", "us", "ns"):
            raise ValueError(
                f"Invalid time_unit; expected one of {{'ns','us','ms'}}, got {time_unit!r}"
            )

        self.time_unit = time_unit

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if isclass(other) and issubclass(other, Duration):
            return True
        elif isinstance(other, Duration):
            return self.time_unit == other.time_unit
        else:
            return False

    def __hash__(self) -> int:
        return hash((Duration, self.time_unit))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(time_unit={self.time_unit!r})"


class Categorical(DataType):
    """A categorical encoding of a set of strings."""


class Object(DataType):
    """Type for wrapping arbitrary Python objects."""


class Null(DataType):
    """Type representing Null / None values."""


class Unknown(DataType):
    """Type representing Datatype values that could not be determined statically."""


class List(NestedType):
    def __init__(self, inner: PolarsDataType | PythonDataType = Float64):
        """
        Nested list/array type.

        Parameters
        ----------
        inner
            The `DataType` of values within the list

        """
        self.inner = dt.py_type_to_dtype(inner)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # The comparison allows comparing objects to classes
        # and specific inner types to none specific.
        # if one of the arguments is not specific about its inner type
        # we infer it as being equal.
        # List[i64] == List[i64] == True
        # List[i64] == List == True
        # List[i64] == List[None] == True
        # List[i64] == List[f32] == False

        # allow comparing object instances to class
        if isclass(other) and issubclass(other, List):
            return True
        elif isinstance(other, List):
            return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash((List, self.inner))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r})"


class Struct(NestedType):
    def __init__(self, fields: Sequence[Field] | SchemaDict | None = None):
        """
        Struct composite type.

        Parameters
        ----------
        fields
            The sequence of fields that make up the struct

        """
        if fields is None:
            self.fields = [Field("", Null)]
        elif isinstance(fields, Mapping):
            self.fields = [Field(name, dtype) for name, dtype in fields.items()]
        else:
            self.fields = list(fields)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # The comparison allows comparing objects to classes, and specific
        # inner types to those without (eg: inner=None). if one of the
        # arguments is not specific about its inner type we infer it
        # as being equal. (See the List type for more info).
        if isclass(other) and issubclass(other, Struct):
            return True
        elif isinstance(other, Struct):
            return any((f is None) for f in (self.fields, other.fields)) or (
                self.fields == other.fields
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(Struct)

    def __iter__(self) -> Iterator[tuple[str, PolarsDataType]]:
        for fld in self.fields or []:
            yield fld.name, fld.dtype

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.fields})"

    def to_schema(self) -> SchemaDict | None:
        """Return Struct dtype as a schema dict."""
        return dict(self)


class Field:
    def __init__(self, name: str, dtype: PolarsDataType):
        """
        Definition of a single field within a `Struct` DataType.

        Parameters
        ----------
        name
            The name of the field within its parent `Struct`
        dtype
            The `DataType` of the field's values

        """
        self.name = name
        self.dtype = dt.py_type_to_dtype(dtype)

    def __eq__(self, other: Field) -> bool:  # type: ignore[override]
        return (self.name == other.name) & (self.dtype == other.dtype)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.name!r}, {self.dtype})"
