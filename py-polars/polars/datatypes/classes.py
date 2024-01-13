from __future__ import annotations

import contextlib
from collections import OrderedDict
from datetime import timezone
from inspect import isclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import polars.datatypes

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import dtype_str_repr as _dtype_str_repr

if TYPE_CHECKING:
    from polars.type_aliases import (
        CategoricalOrdering,
        PolarsDataType,
        PythonDataType,
        SchemaDict,
        TimeUnit,
    )


class classinstmethod(classmethod):  # type: ignore[type-arg]
    """Decorator that allows a method to be called from the class OR instance."""

    def __get__(self, instance: Any, type_: type) -> Any:  # type: ignore[override]
        get = super().__get__ if instance is None else self.__func__.__get__
        return get(instance, type_)


class DataTypeClass(type):
    """Metaclass for nicely printing DataType classes."""

    def __repr__(cls) -> str:
        return cls.__name__

    def _string_repr(cls) -> str:
        return _dtype_str_repr(cls)

    # Methods below defined here in signature only to satisfy mypy

    @classmethod
    def base_type(cls) -> DataTypeClass:  # noqa: D102
        ...

    @classmethod
    def is_(cls, other: PolarsDataType) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_not(cls, other: PolarsDataType) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_numeric(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_decimal(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_integer(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_signed_integer(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_unsigned_integer(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_float(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_temporal(cls) -> bool:  # noqa: D102
        ...

    @classmethod
    def is_nested(self) -> bool:  # noqa: D102
        ...


class DataType(metaclass=DataTypeClass):
    """The The base class for all Polars data types."""

    def __reduce__(self) -> Any:
        return (_custom_reconstruct, (type(self), object, None), self.__dict__)

    def _string_repr(self) -> str:
        return _dtype_str_repr(self)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        if type(other) is DataTypeClass:
            return issubclass(other, type(self))
        else:
            return isinstance(other, type(self))

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __repr__(self) -> str:
        return self.__class__.__name__

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

    @classinstmethod  # type: ignore[arg-type]
    def is_(self, other: PolarsDataType) -> bool:
        """
        Check if this DataType is the same as another DataType.

        This is a stricter check than `self == other`, as it enforces an exact
        match of all dtype attributes for nested and/or uninitialised dtypes.

        Parameters
        ----------
        other
            The other polars dtype to compare with.

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

        .. deprecated:: 0.19.14
            Use `not dtype.is_(...)` instead.

        This is a stricter check than `self != other`, as it enforces an exact
        match of all dtype attributes for nested and/or uninitialised dtypes.

        Parameters
        ----------
        other
            The other polars dtype to compare with.

        Examples
        --------
        >>> pl.List != pl.List(pl.Int32)
        False
        >>> pl.List.is_not(pl.List(pl.Int32))  # doctest: +SKIP
        True

        """
        from polars.utils.deprecation import issue_deprecation_warning

        issue_deprecation_warning(
            "`DataType.is_not` is deprecated and will be removed in the next breaking release."
            " Use `not dtype.is_(...)` instead.",
            version="0.19.14",
        )
        return not self.is_(other)

    @classmethod
    def is_numeric(cls) -> bool:
        """Check whether the data type is a numeric type."""
        return issubclass(cls, NumericType)

    @classmethod
    def is_decimal(cls) -> bool:
        """Check whether the data type is a decimal type."""
        return issubclass(cls, Decimal)

    @classmethod
    def is_integer(cls) -> bool:
        """Check whether the data type is an integer type."""
        return issubclass(cls, IntegerType)

    @classmethod
    def is_signed_integer(cls) -> bool:
        """Check whether the data type is a signed integer type."""
        return issubclass(cls, SignedIntegerType)

    @classmethod
    def is_unsigned_integer(cls) -> bool:
        """Check whether the data type is an unsigned integer type."""
        return issubclass(cls, UnsignedIntegerType)

    @classmethod
    def is_float(cls) -> bool:
        """Check whether the data type is a temporal type."""
        return issubclass(cls, FloatType)

    @classmethod
    def is_temporal(cls) -> bool:
        """Check whether the data type is a temporal type."""
        return issubclass(cls, TemporalType)

    @classmethod
    def is_nested(cls) -> bool:
        """Check whether the data type is a nested type."""
        return issubclass(cls, NestedType)


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
    """A group of data types."""

    _match_base_type: bool

    def __new__(
        cls, items: Iterable[DataType | DataTypeClass], *, match_base_type: bool = True
    ) -> DataTypeGroup:
        """
        Construct a DataTypeGroup.

        Parameters
        ----------
        items :
            An iterable of data types.
        match_base_type:
            Whether to match the base type.

        """
        for it in items:
            if not isinstance(it, (DataType, DataTypeClass)):
                raise TypeError(
                    f"DataTypeGroup items must be dtypes; found {type(it).__name__!r}"
                )
        dtype_group = super().__new__(cls, items)  # type: ignore[arg-type]
        dtype_group._match_base_type = match_base_type
        return dtype_group

    def __contains__(self, item: Any) -> bool:
        if self._match_base_type and isinstance(item, (DataType, DataTypeClass)):
            item = item.base_type()
        return super().__contains__(item)


class NumericType(DataType):
    """The base class for numeric data types."""


class IntegerType(NumericType):
    """The base class for integer data types."""


class SignedIntegerType(IntegerType):
    """The base class for signed integer data types."""


class UnsignedIntegerType(IntegerType):
    """The base class for unsigned integer data types."""


class FloatType(NumericType):
    """The base class for float data types."""


class TemporalType(DataType):
    """The base class for temporal data types."""


class NestedType(DataType):
    """The base class for nested data types."""


class Int8(SignedIntegerType):
    """An 8-bit signed integer type."""


class Int16(SignedIntegerType):
    """A 16-bit signed integer type."""


class Int32(SignedIntegerType):
    """A 32-bit signed integer type."""


class Int64(SignedIntegerType):
    """A 64-bit signed integer type."""


class UInt8(UnsignedIntegerType):
    """An 8-bit unsigned integer type."""


class UInt16(UnsignedIntegerType):
    """A 16-bit unsigned integer type."""


class UInt32(UnsignedIntegerType):
    """A 32-bit unsigned integer type."""


class UInt64(UnsignedIntegerType):
    """A 64-bit unsigned integer type."""


class Float32(FloatType):
    """A 32-bit floating point type."""


class Float64(FloatType):
    """A 64-bit floating point type."""


class Decimal(NumericType):
    """
    A Decimal 128-bit type with an optional precision and non-negative scale.

    .. warning::
        This is an experimental work-in-progress feature and may not work as expected.

    """

    precision: int | None
    scale: int

    def __init__(
        self,
        precision: int | None = None,
        scale: int = 0,
    ):
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
    """A boolean (`True`/`False`) type."""


class String(DataType):
    """A UTF-8 encoded string type."""


# Allow Utf8 as an alias for String
Utf8 = String


class Binary(DataType):
    """A binary type, analogous to Python's built-in `bytes` type."""


class Date(TemporalType):
    """A calendar date typem, analogous to Python's `datetime.date` type."""


class Time(TemporalType):
    """A time-of-day type, analogous to Python's `datetime.time` type."""


class Datetime(TemporalType):
    """A calendar date and time type, analogous to Python's `datetime.datetime` type."""

    time_unit: TimeUnit | None = None
    time_zone: str | None = None

    def __init__(
        self, time_unit: TimeUnit | None = "us", time_zone: str | timezone | None = None
    ):
        """
        A calendar date and time type, analogous to Python's `datetime.datetime` type.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            The time unit (precision).
        time_zone
            A time zone string, as defined in `zoneinfo
            <https://docs.python.org/3/library/zoneinfo.html>_. (Run
            `import zoneinfo; zoneinfo.available_timezones()` for the full list of valid
            strings. When instantiating a `Datetime` object for the purpose of matching
            dtypes, you can use "*" to check for `Datetime` columns that have any
            timezone.

        """
        if isinstance(time_zone, timezone):
            time_zone = str(time_zone)

        self.time_unit = time_unit or "us"
        self.time_zone = time_zone

        if self.time_unit not in ("ms", "us", "ns"):
            raise ValueError(
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms'}}, got {self.time_unit!r}."
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
    """A time duration type, analogous to Python's `datetime.timedelta` type."""

    time_unit: TimeUnit | None = None

    def __init__(self, time_unit: TimeUnit = "us"):
        """
        A time duration type, analogous to Python's `datetime.timedelta` type.

        Parameters
        ----------
        time_unit : {'ns', 'us', 'ms'}
            The time unit (precision).

        """
        self.time_unit = time_unit
        if self.time_unit not in ("ms", "us", "ns"):
            raise ValueError(
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms'}}, got {self.time_unit!r}."
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
    """
    A categorical encoding of a set of strings.

    Parameters
    ----------
        ordering : {'lexical', 'physical'}
            Whether the ordering is by order of appearance (`'physical'`, the default)
            or alphabetical (`'lexical'`).

    """

    ordering: CategoricalOrdering | None

    def __init__(
        self,
        ordering: CategoricalOrdering | None = "physical",
    ):
        self.ordering = ordering

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ordering={self.ordering!r})"

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Categorical):
            return True
        elif isinstance(other, Categorical):
            return self.ordering == other.ordering
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.ordering))


class Enum(DataType):
    """
    A categorical encoding of a fixed set of strings with a user-specified ordering.

    .. warning::
        This is an experimental work-in-progress feature and may not work as expected.

    """

    categories: list[str]

    def __init__(self, categories: Iterable[str]):
        """
        A categorical encoding of a fixed set of strings with a user-specified ordering.

        Parameters
        ----------
        categories
            The strings forming the `Enum`, in order.

        """
        if not isinstance(categories, list):
            categories = list(categories)

        seen: set[str] = set()
        for cat in categories:
            if cat in seen:
                raise ValueError(
                    f"Enum categories must be unique; found duplicate {cat!r}"
                )
            if not isinstance(cat, str):
                raise TypeError(
                    f"Enum categories must be strings; found {cat!r} ({type(cat).__name__})"
                )
            seen.add(cat)

        self.categories = categories

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Enum):
            return True
        elif isinstance(other, Enum):
            return self.categories == other.categories
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self.categories)))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(categories={self.categories!r})"


class Object(DataType):
    """A type for wrapping arbitrary Python objects."""


class Null(DataType):
    """A type representing `null` / `None` values."""


class Unknown(DataType):
    """A type representing dtypes that could not be determined statically."""


class List(NestedType):
    """A variable-length list type."""

    inner: PolarsDataType | None = None

    def __init__(self, inner: PolarsDataType | PythonDataType):
        """
        A variable-length list type.

        Parameters
        ----------
        inner
            The `DataType` of the values within each list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "integer_lists": [[1, 2], [3, 4]],
        ...         "float_lists": [[1.0, 2.0], [3.0, 4.0]],
        ...     }
        ... )
        >>> df
        shape: (2, 2)
        ┌───────────────┬─────────────┐
        │ integer_lists ┆ float_lists │
        │ ---           ┆ ---         │
        │ list[i64]     ┆ list[f64]   │
        ╞═══════════════╪═════════════╡
        │ [1, 2]        ┆ [1.0, 2.0]  │
        │ [3, 4]        ┆ [3.0, 4.0]  │
        └───────────────┴─────────────┘

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
    """A fixed-length list type."""

    inner: PolarsDataType | None = None
    width: int

    def __init__(self, inner: PolarsDataType | PythonDataType, width: int):
        """
        A fixed-length list type.

        Parameters
        ----------
        inner
            The `DataType` of the values within each array.
        width
            The length of the lists.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
        >>> s
        shape: (2,)
        Series: 'a' [array[i64, 2]]
        [
                [1, 2]
                [4, 3]
        ]

        """
        self.inner = polars.datatypes.py_type_to_dtype(inner)
        self.width = width

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # This equality check allows comparison of type classes and type instances.
        # If a parent type is not specific about its inner type, we infer it as equal:
        # > array[i64] == array[i64] -> True
        # > array[i64] == array[f32] -> False
        # > array[i64] == array      -> True

        # allow comparing object instances to class
        if type(other) is DataTypeClass and issubclass(other, Array):
            return True
        if isinstance(other, Array):
            if self.width != other.width:
                return False
            elif self.inner is None or other.inner is None:
                return True
            else:
                return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner, self.width))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r}, width={self.width})"


class Field:
    """A single field within a :class:`Struct` `DataType`."""

    def __init__(self, name: str, dtype: PolarsDataType):
        """
        A single field within a :class:`Struct` `DataType`.

        Parameters
        ----------
        name
            The name of the field within its parent :class:`Struct`.
        dtype
            The `DataType` of the field's values.

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
    """A structured composite type with keys and values."""

    fields: list[Field]

    def __init__(self, fields: Sequence[Field] | SchemaDict):
        """
        A structured composite type with keys and values.

        Parameters
        ----------
        fields
            The fields that make up the struct. Can be either a sequence of
            :class:`Field` objects or a mapping of column names to data types.

        Examples
        --------
        Initialize using a dictionary:

        >>> dtype = pl.Struct({"a": pl.Int8, "b": pl.List(pl.String)})
        >>> dtype
        Struct({'a': Int8, 'b': List(String)})

        Initialize using a list of `Field` objects:

        >>> dtype = pl.Struct(
        ...     [pl.Field("a", pl.Int8), pl.Field("b", pl.List(pl.String))]
        ... )
        >>> dtype
        Struct({'a': Int8, 'b': List(String)})

        When initializing a `Series`, Polars can infer the fields' dtypes from the data.

        >>> s = pl.Series([{"a": 1, "b": ["x", "y"]}, {"a": 2, "b": ["z"]}])
        >>> s
        shape: (2,)
        Series: '' [struct[2]]
        [
                {1,["x", "y"]}
                {2,["z"]}
        ]
        >>> s.dtype
        Struct({'a': Int64, 'b': List(String)})
        """
        if isinstance(fields, Mapping):
            self.fields = [Field(name, dtype) for name, dtype in fields.items()]
        else:
            self.fields = list(fields)

    def __eq__(self, other: PolarsDataType) -> bool:  # type: ignore[override]
        # The comparison allows comparing objects to classes, and specific
        # inner types to those without (e.g. inner=None). if one of the
        # arguments is not specific about its inner type we infer it
        # as being equal. (See the List type for more info).
        if isclass(other) and issubclass(other, Struct):
            return True
        elif isinstance(other, Struct):
            return self.fields == other.fields
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self.fields)))

    def __iter__(self) -> Iterator[tuple[str, PolarsDataType]]:
        for fld in self.fields:
            yield fld.name, fld.dtype

    def __reversed__(self) -> Iterator[tuple[str, PolarsDataType]]:
        for fld in reversed(self.fields):
            yield fld.name, fld.dtype

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({dict(self)})"

    def to_schema(self) -> OrderedDict[str, PolarsDataType]:
        """Return Struct dtype as a schema dict."""
        return OrderedDict(self)
