from __future__ import annotations

import contextlib
from datetime import timezone
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

import polars.datatypes

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import dtype_str_repr as _dtype_str_repr

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType, PythonDataType, SchemaDict, TimeUnit


class classinstmethod(classmethod):  # type: ignore[type-arg]
    """Decorator that allows a method to be called from the class OR instance."""

    def __get__(self, instance: Any, type_: type) -> Any:  # type: ignore[override]
        get = super().__get__ if instance is None else self.__func__.__get__
        return get(instance, type_)


class classproperty:
    """Equivalent to @property, but works on a class (doesn't require an instance)."""

    def __init__(self, method: Callable[..., Any] | None = None) -> None:
        self.fget = method

    def __get__(self, instance: Any, cls: type | None = None) -> Any:
        return self.fget(cls)  # type: ignore[misc]

    def getter(self, method: Callable[..., Any]) -> Any:
        self.fget = method
        return self


class DataTypeClass(type):
    """Metaclass for nicely printing DataType classes."""

    def __repr__(cls) -> str:
        return cls.__name__

    def _string_repr(cls) -> str:
        return _dtype_str_repr(cls)

    def base_type(cls) -> PolarsDataType:
        return cls

    @classproperty
    def is_nested(self) -> bool:
        return False

    @classmethod
    def is_(cls, other: PolarsDataType) -> bool:
        return cls == other and hash(cls) == hash(other)

    @classmethod
    def is_not(cls, other: PolarsDataType) -> bool:
        return not cls.is_(other)


class DataType(metaclass=DataTypeClass):
    """Base class for all Polars data types."""

    def __new__(cls, *args: Any, **kwargs: Any) -> PolarsDataType:  # type: ignore[misc]
        # this formulation allows for equivalent use of "pl.Type" and "pl.Type()", while
        # still respecting types that take initialisation params (eg: Duration/Datetime)
        if args or kwargs:
            return super().__new__(cls)
        return cls

    def __reduce__(self) -> Any:
        return (_custom_reconstruct, (type(self), object, None), self.__dict__)

    def _string_repr(self) -> str:
        return _dtype_str_repr(self)

    @classmethod
    def base_type(cls) -> DataTypeClass:
        """
        Return this DataType's fundamental/root type class.

        Examples
        --------
        >>> pl.Datetime("ns").base_type()
        Datetime
        >>> pl.List(pl.Int32).base_type()
        List
        >>> pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)]).base_type()
        Struct
        """
        return cls

    @classproperty
    def is_nested(self) -> bool:
        return False

    @classinstmethod  # type: ignore[arg-type]
    def is_(self, other: PolarsDataType) -> bool:
        """
        Check if this DataType is the same as another DataType.

        This is a stricter check than ``self == other``, as it enforces an exact
        match of all dtype attributes for nested and/or uninitialised dtypes.

        Parameters
        ----------
        other
            the other polars dtype to compare with.

        Examples
        --------
        >>> pl.List == pl.List(pl.Int32)
        True
        >>> pl.List.is_(pl.List(pl.Int32))
        False

        """
        return self == other and hash(self) == hash(other)

    @classinstmethod  # type: ignore[arg-type]
    def is_not(self, other: PolarsDataType) -> bool:
        """
        Check if this DataType is NOT the same as another DataType.

        This is a stricter check than ``self != other``, as it enforces an exact
        match of all dtype attributes for nested and/or uninitialised dtypes.

        Parameters
        ----------
        other
            the other polars dtype to compare with.

        Examples
        --------
        >>> pl.List != pl.List(pl.Int32)
        False
        >>> pl.List.is_not(pl.List(pl.Int32))
        True

        """
        return not self.is_(other)


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


class DataTypeGroup(frozenset):  # type: ignore[type-arg]
    _match_base_type: bool

    def __new__(cls, items: Any, *, match_base_type: bool = True) -> DataTypeGroup:
        for it in items:
            if not isinstance(it, (DataType, DataTypeClass)):
                raise TypeError(
                    f"DataTypeGroup items must be dtypes; found {type(it).__name__!r}"
                )
        dtype_group = super().__new__(cls, items)
        dtype_group._match_base_type = match_base_type
        return dtype_group

    def __contains__(self, item: Any) -> bool:
        if self._match_base_type and isinstance(item, (DataType, DataTypeClass)):
            item = item.base_type()
        return super().__contains__(item)


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

    @classproperty
    def is_nested(self) -> bool:
        return True


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

    precision: int | None
    scale: int

    def __init__(self, precision: int | None, scale: int):
        self.precision = precision
        self.scale = scale

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(precision={self.precision}, scale={self.scale})"
        )

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Decimal):
            return True
        elif isinstance(other, Decimal):
            return self.precision == other.precision and self.scale == other.scale
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.precision, self.scale))


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

    time_unit: TimeUnit | None = None
    time_zone: str | None = None

    def __init__(
        self, time_unit: TimeUnit | None = "us", time_zone: str | timezone | None = None
    ):
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
        if isinstance(time_zone, timezone):
            time_zone = str(time_zone)

        self.time_unit = time_unit or "us"
        self.time_zone = time_zone

        if self.time_unit not in ("ms", "us", "ns"):
            raise ValueError(
                f"Invalid time_unit; expected one of {{'ns','us','ms'}}, got {self.time_unit!r}"
            )

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Datetime):
            return True
        elif isinstance(other, Datetime):
            return (
                self.time_unit == other.time_unit and self.time_zone == other.time_zone
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.time_unit, self.time_zone))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(time_unit={self.time_unit!r}, time_zone={self.time_zone!r})"
        )


class Duration(TemporalType):
    """Time duration/delta type."""

    time_unit: TimeUnit | None = None

    def __init__(self, time_unit: TimeUnit = "us"):
        """
        Time duration/delta type.

        Parameters
        ----------
        time_unit : {'us', 'ns', 'ms'}
            Unit of time.

        """
        self.time_unit = time_unit
        if self.time_unit not in ("ms", "us", "ns"):
            raise ValueError(
                f"Invalid time_unit; expected one of {{'ns','us','ms'}}, got {self.time_unit!r}"
            )

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Duration):
            return True
        elif isinstance(other, Duration):
            return self.time_unit == other.time_unit
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.time_unit))

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
    inner: PolarsDataType | None = None

    def __init__(self, inner: PolarsDataType | PythonDataType):
        """
        Nested list/array type.

        Parameters
        ----------
        inner
            The `DataType` of values within the list

        """
        self.inner = polars.datatypes.py_type_to_dtype(inner)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # This equality check allows comparison of type classes and type instances.
        # If a parent type is not specific about its inner type, we infer it as equal:
        # > list[i64] == list[i64] -> True
        # > list[i64] == list[f32] -> False
        # > list[i64] == list      -> True

        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, List):
            return True
        if isinstance(other, List):
            if self.inner is None or other.inner is None:
                return True
            else:
                return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r})"


class Array(NestedType):
    inner: PolarsDataType | None = None
    width: int

    def __init__(self, width: int, inner: PolarsDataType | PythonDataType = Null):
        """
        Nested list/array type.

        Parameters
        ----------
        width
            The fixed size length of the inner arrays.
        inner
            The `DataType` of values within the list

        """
        self.width = width
        self.inner = polars.datatypes.py_type_to_dtype(inner)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # This equality check allows comparison of type classes and type instances.
        # If a parent type is not specific about its inner type, we infer it as equal:
        # > fixed-size-list[i64] == fixed-size-list[i64] -> True
        # > fixed-size-list[i64] == fixed-size-list[f32] -> False
        # > fixed-size-list[i64] == fixed-size-list      -> True

        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Array):
            return True
        if isinstance(other, Array):
            if self.inner is None or other.inner is None:
                return True
            else:
                return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r})"


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
        self.dtype = polars.datatypes.py_type_to_dtype(dtype)

    def __eq__(self, other: Field) -> bool:  # type: ignore[override]
        return (self.name == other.name) & (self.dtype == other.dtype)

    def __hash__(self) -> int:
        return hash((self.name, self.dtype))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.name!r}, {self.dtype})"


class Struct(NestedType):
    def __init__(self, fields: Sequence[Field] | SchemaDict):
        """
        Struct composite type.

        Parameters
        ----------
        fields
            The sequence of fields that make up the struct

        """
        if isinstance(fields, Mapping):
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
        return hash((self.__class__, tuple(self.fields)))

    def __iter__(self) -> Iterator[tuple[str, PolarsDataType]]:
        for fld in self.fields or []:
            yield fld.name, fld.dtype

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.fields})"

    def to_schema(self) -> SchemaDict | None:
        """Return Struct dtype as a schema dict."""
        return dict(self)
