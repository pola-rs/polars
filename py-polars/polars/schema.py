from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
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
        schema: Mapping[str, DataType] | Iterable[tuple[str, DataType]] | None = None,
    ):
        schema = schema or {}
        super().__init__(schema)

    def names(self) -> list[str]:
        """Get the column names of the schema."""
        return list(self.keys())

    def dtypes(self) -> list[DataType]:
        """Get the data types of the schema."""
        return list(self.values())

    def len(self) -> int:
        """Get the number of columns in the schema."""
        return len(self)
