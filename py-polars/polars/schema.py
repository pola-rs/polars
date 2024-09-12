from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable

from polars.datatypes._parse import parse_into_dtype

if TYPE_CHECKING:
    from polars._typing import PythonDataType
    from polars.datatypes import DataType

    BaseSchema = OrderedDict[str, DataType]
else:
    # Python 3.8 does not support generic OrderedDict at runtime
    BaseSchema = OrderedDict

__all__ = ["Schema"]


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
            Mapping[str, DataType | PythonDataType]
            | Iterable[tuple[str, DataType | PythonDataType]]
            | None
        ) = None,
    ) -> None:
        input = (
            schema.items() if schema and isinstance(schema, Mapping) else (schema or {})
        )
        super().__init__({name: parse_into_dtype(tp) for name, tp in input})  # type: ignore[misc]

    def __setitem__(self, name: str, dtype: DataType | PythonDataType) -> None:
        super().__setitem__(name, parse_into_dtype(dtype))  # type: ignore[assignment]

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
        Return Schema as a dictionary of column names and their Python types.

        Examples
        --------
        >>> s = pl.Schema({"x": pl.Int8(), "y": pl.String(), "z": pl.Duration("ms")})
        >>> s.to_python()
        {'x': <class 'int'>, 'y':  <class 'str'>, 'z': <class 'datetime.timedelta'>}
        """
        return {name: tp.to_python() for name, tp in self.items()}
