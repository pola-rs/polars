from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

from polars._typing import PythonDataType
from polars.datatypes import DataType, DataTypeClass, is_polars_dtype
from polars.datatypes._parse import parse_into_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias


if sys.version_info >= (3, 10):

    def _required_init_args(tp: DataTypeClass) -> bool:
        # note: this check is ~20% faster than the check for a
        # custom "__init__", below, but is not available on py39
        return bool(tp.__annotations__)
else:

    def _required_init_args(tp: DataTypeClass) -> bool:
        # indicates override of the default __init__
        # (eg: this type requires specific args)
        return "__init__" in tp.__dict__


BaseSchema = OrderedDict[str, DataType]
SchemaInitDataType: TypeAlias = Union[DataType, DataTypeClass, PythonDataType]


__all__ = ["Schema"]


def _check_dtype(tp: DataType | DataTypeClass) -> DataType:
    if not isinstance(tp, DataType):
        # note: if nested/decimal, or has signature params, this implies required args
        if tp.is_nested() or tp.is_decimal() or _required_init_args(tp):
            msg = f"dtypes must be fully-specified, got: {tp!r}"
            raise TypeError(msg)
        tp = tp()
    return tp  # type: ignore[return-value]


class Schema(BaseSchema):
    """
    Ordered mapping of column names to their data type.

    Parameters
    ----------
    schema
        The schema definition given by column names and their associated *instantiated*
        Polars data type. Accepts a mapping or an iterable of tuples.

    Examples
    --------
    Define a schema by passing *instantiated* data types.

    >>> schema = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})
    >>> schema
    Schema({'foo': Int8, 'bar': String})

    Access the data type associated with a specific column name.

    >>> schema["foo"]
    Int8

    Access various schema properties using the `names`, `dtypes`, and `len` methods.

    >>> schema.names()
    ['foo', 'bar']
    >>> schema.dtypes()
    [Int8, String]
    >>> schema.len()
    2
    """

    def __init__(
        self,
        schema: (
            Mapping[str, SchemaInitDataType]
            | Iterable[tuple[str, SchemaInitDataType]]
            | None
        ) = None,
        *,
        check_dtypes: bool = True,
    ) -> None:
        input = (
            schema.items() if schema and isinstance(schema, Mapping) else (schema or {})
        )
        for name, tp in input:  # type: ignore[misc]
            if not check_dtypes:
                super().__setitem__(name, tp)  # type: ignore[assignment]
            elif is_polars_dtype(tp):
                super().__setitem__(name, _check_dtype(tp))
            else:
                self[name] = tp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        if len(self) != len(other):
            return False
        for (nm1, tp1), (nm2, tp2) in zip(self.items(), other.items()):
            if nm1 != nm2 or not tp1.is_(tp2):
                return False
        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __setitem__(
        self, name: str, dtype: DataType | DataTypeClass | PythonDataType
    ) -> None:
        dtype = _check_dtype(parse_into_dtype(dtype))
        super().__setitem__(name, dtype)

    def names(self) -> list[str]:
        """Get the column names of the schema."""
        return list(self.keys())

    def dtypes(self) -> list[DataType]:
        """Get the data types of the schema."""
        return list(self.values())

    def len(self) -> int:
        """Get the number of columns in the schema."""
        return len(self)

    def to_python(self) -> dict[str, type]:
        """
        Return a dictionary of column names and Python types.

        Examples
        --------
        >>> s = pl.Schema({"x": pl.Int8(), "y": pl.String(), "z": pl.Duration("ms")})
        >>> s.to_python()
        {'x': <class 'int'>, 'y':  <class 'str'>, 'z': <class 'datetime.timedelta'>}
        """
        return {name: tp.to_python() for name, tp in self.items()}
