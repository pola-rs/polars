from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from datetime import date, datetime, time, timedelta
    from decimal import Decimal
    from typing import TypeAlias

    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession
    from sqlalchemy.orm import Session

    from polars import DataFrame, Expr, LazyFrame, Series
    from polars._dependencies import numpy as np
    from polars._dependencies import pandas as pd
    from polars._dependencies import pyarrow as pa
    from polars._dependencies import torch
    from polars.datatypes import DataType, DataTypeClass, IntegerType, TemporalType
    from polars.lazyframe.engine_config import GPUEngine
    from polars.selectors import Selector


class ArrowArrayExportable(Protocol):
    """Type protocol for Arrow C Data Interface via Arrow PyCapsule Interface."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...


class ArrowStreamExportable(Protocol):
    """Type protocol for Arrow C Stream Interface via Arrow PyCapsule Interface."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


class ArrowSchemaExportable(Protocol):
    """Type protocol for Arrow C Schema Interface via Arrow PyCapsule Interface."""

    def __arrow_c_schema__(self) -> object: ...


# Data types
PolarsDataType: TypeAlias = Union["DataTypeClass", "DataType"]
PolarsTemporalType: TypeAlias = Union[type["TemporalType"], "TemporalType"]
PolarsIntegerType: TypeAlias = Union[type["IntegerType"], "IntegerType"]
OneOrMoreDataTypes: TypeAlias = PolarsDataType | Iterable[PolarsDataType]
PythonDataType: TypeAlias = (
    type[int]
    | type[float]
    | type[bool]
    | type[str]
    | type["date"]
    | type["time"]
    | type["datetime"]
    | type["timedelta"]
    | type[list[Any]]
    | type[tuple[Any, ...]]
    | type[bytes]
    | type[object]
    | type["Decimal"]
    | type[None]
)

SchemaDefinition: TypeAlias = (
    Mapping[str, PolarsDataType | PythonDataType | None]
    | Sequence[str | tuple[str, PolarsDataType | PythonDataType | None]]
)
SchemaDict: TypeAlias = Mapping[str, PolarsDataType]

NumericLiteral: TypeAlias = Union[int, float, "Decimal"]
TemporalLiteral: TypeAlias = Union["date", "time", "datetime", "timedelta"]
NonNestedLiteral: TypeAlias = NumericLiteral | TemporalLiteral | str | bool | bytes
# Python literal types (can convert into a `lit` expression)
PythonLiteral: TypeAlias = Union[NonNestedLiteral, "np.ndarray[Any, Any]", list[Any]]
# Inputs that can convert into a `col` expression
IntoExprColumn: TypeAlias = Union["Expr", "Series", str]
# Inputs that can convert into an expression
IntoExpr: TypeAlias = PythonLiteral | IntoExprColumn | None

ComparisonOperator: TypeAlias = Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"]

# selector type, and related collection/sequence
SelectorType: TypeAlias = "Selector"
ColumnNameOrSelector: TypeAlias = Union["str", SelectorType]

# User-facing string literal types
# The following all have an equivalent Rust enum with the same name
Ambiguous: TypeAlias = Literal["earliest", "latest", "raise", "null"]
AvroCompression: TypeAlias = Literal["uncompressed", "snappy", "deflate"]
CsvQuoteStyle: TypeAlias = Literal["necessary", "always", "non_numeric", "never"]
CategoricalOrdering: TypeAlias = Literal["physical", "lexical"]
CsvEncoding: TypeAlias = Literal["utf8", "utf8-lossy"]
ColumnMapping: TypeAlias = tuple[
    Literal["iceberg-column-mapping"],
    # This is "pa.Schema". Not typed as that causes pyright strict type checking
    # failures for users who don't have pyarrow-stubs installed.
    Any,
]
DefaultFieldValues: TypeAlias = tuple[
    Literal["iceberg"], dict[int, Union["Series", str]]
]
DeletionFiles: TypeAlias = tuple[
    Literal["iceberg-position-delete"], dict[int, list[str]]
]
FillNullStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
FloatFmt: TypeAlias = Literal["full", "mixed"]
IndexOrder: TypeAlias = Literal["c", "fortran"]
IpcCompression: TypeAlias = Literal["uncompressed", "lz4", "zstd"]
JoinValidation: TypeAlias = Literal["m:m", "m:1", "1:m", "1:1"]
Label: TypeAlias = Literal["left", "right", "datapoint"]
MaintainOrderJoin: TypeAlias = Literal[
    "none", "left", "right", "left_right", "right_left"
]
NdjsonCompression: TypeAlias = Literal["uncompressed", "gzip", "zstd"]
NonExistent: TypeAlias = Literal["raise", "null"]
NullBehavior: TypeAlias = Literal["ignore", "drop"]
ParallelStrategy: TypeAlias = Literal[
    "auto", "columns", "row_groups", "prefiltered", "none"
]
ParquetCompression: TypeAlias = Literal[
    "lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"
]
PivotAgg: TypeAlias = Literal[
    "min", "max", "first", "last", "sum", "mean", "median", "len", "item"
]
QuantileMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
]
RankMethod: TypeAlias = Literal["average", "min", "max", "dense", "ordinal", "random"]
Roll: TypeAlias = Literal["raise", "forward", "backward"]
RoundMode: TypeAlias = Literal["half_to_even", "half_away_from_zero"]
SerializationFormat: TypeAlias = Literal["binary", "json"]
Endianness: TypeAlias = Literal["little", "big"]
SizeUnit: TypeAlias = Literal[
    "b",
    "kb",
    "mb",
    "gb",
    "tb",
    "bytes",
    "kilobytes",
    "megabytes",
    "gigabytes",
    "terabytes",
]
StartBy: TypeAlias = Literal[
    "window",
    "datapoint",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
SyncOnCloseMethod: TypeAlias = Literal["data", "all"]
TimeUnit: TypeAlias = Literal["ns", "us", "ms"]
UnicodeForm: TypeAlias = Literal["NFC", "NFKC", "NFD", "NFKD"]
UniqueKeepStrategy: TypeAlias = Literal["first", "last", "any", "none"]
UnstackDirection: TypeAlias = Literal["vertical", "horizontal"]
MapElementsStrategy: TypeAlias = Literal["thread_local", "threading"]

# The following have a Rust enum equivalent with a different name
AsofJoinStrategy: TypeAlias = Literal["backward", "forward", "nearest"]  # AsofStrategy
ClosedInterval: TypeAlias = Literal["left", "right", "both", "none"]  # ClosedWindow
InterpolationMethod: TypeAlias = Literal["linear", "nearest"]
JoinStrategy: TypeAlias = Literal[
    "inner", "left", "right", "full", "semi", "anti", "cross", "outer"
]  # JoinType
ListToStructWidthStrategy: TypeAlias = Literal["first_non_null", "max_width"]

# The following have no equivalent on the Rust side
ConcatMethod = Literal[
    "vertical",
    "vertical_relaxed",
    "diagonal",
    "diagonal_relaxed",
    "horizontal",
    "align",
    "align_full",
    "align_inner",
    "align_left",
    "align_right",
]
CorrelationMethod: TypeAlias = Literal["pearson", "spearman"]
DbReadEngine: TypeAlias = Literal["adbc", "connectorx"]
DbWriteEngine: TypeAlias = Literal["sqlalchemy", "adbc"]
DbWriteMode: TypeAlias = Literal["replace", "append", "fail"]
EpochTimeUnit = Literal["ns", "us", "ms", "s", "d"]
JaxExportType: TypeAlias = Literal["array", "dict"]
Orientation: TypeAlias = Literal["col", "row"]
SearchSortedSide: TypeAlias = Literal["any", "left", "right"]
TorchExportType: TypeAlias = Literal["tensor", "dataset", "dict"]
TransferEncoding: TypeAlias = Literal["hex", "base64"]
WindowMappingStrategy: TypeAlias = Literal["group_to_rows", "join", "explode"]
ExplainFormat: TypeAlias = Literal["plain", "tree"]

# type signature for allowed frame init
FrameInitTypes: TypeAlias = Union[
    Mapping[str, Union[Sequence[object], Mapping[str, Sequence[object]], "Series"]],
    Sequence[Any],
    "np.ndarray[Any, Any]",
    "pa.Table",
    "pd.DataFrame",
    "ArrowArrayExportable",
    "ArrowStreamExportable",
    "torch.Tensor",
]

# Excel IO
ColumnFormatDict: TypeAlias = Mapping[
    # dict of colname(s) or selector(s) to format string or dict
    ColumnNameOrSelector | tuple[ColumnNameOrSelector, ...],
    str | Mapping[str, str],
]
ConditionalFormatDict: TypeAlias = Mapping[
    # dict of colname(s) to str, dict, or sequence of str/dict
    ColumnNameOrSelector | Collection[str],
    str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]],
]
ColumnTotalsDefinition: TypeAlias = (
    Mapping[ColumnNameOrSelector | tuple[ColumnNameOrSelector], str]
    | Sequence[str]
    | bool
)
ColumnWidthsDefinition: TypeAlias = (
    Mapping[ColumnNameOrSelector, tuple[str, ...] | int] | int
)
RowTotalsDefinition: TypeAlias = (
    Mapping[str, str | Collection[str]] | Collection[str] | bool
)

# standard/named hypothesis profiles used for parametric testing
ParametricProfileNames: TypeAlias = Literal["fast", "balanced", "expensive"]

# typevars for core polars types
PolarsType = TypeVar("PolarsType", "DataFrame", "LazyFrame", "Series", "Expr")
FrameType = TypeVar("FrameType", "DataFrame", "LazyFrame")
BufferInfo: TypeAlias = tuple[int, int, int]

# type alias for supported spreadsheet engines
ExcelSpreadsheetEngine: TypeAlias = Literal["calamine", "openpyxl", "xlsx2csv"]


class SeriesBuffers(TypedDict):
    """Underlying buffers of a Series."""

    values: Series
    validity: Series | None
    offsets: Series | None


# minimal protocol definitions that can reasonably represent
# an executable connection, cursor, or equivalent object
class BasicConnection(Protocol):
    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        """Return a cursor object."""


class BasicCursor(Protocol):
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a query."""


class Cursor(BasicCursor):
    def fetchall(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch all results."""

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch results in batches."""


AlchemyConnection: TypeAlias = Union["Connection", "Engine", "Session"]
AlchemyAsyncConnection: TypeAlias = Union[
    "AsyncConnection", "AsyncEngine", "AsyncSession"
]
ConnectionOrCursor: TypeAlias = (
    BasicConnection | BasicCursor | Cursor | AlchemyConnection | AlchemyAsyncConnection
)

# Annotations for `__getitem__` methods
SingleIndexSelector: TypeAlias = int
MultiIndexSelector: TypeAlias = Union[
    slice,
    range,
    Sequence[int],
    "Series",
    "np.ndarray[Any, Any]",
]
SingleNameSelector: TypeAlias = str
MultiNameSelector: TypeAlias = Union[
    slice,
    Sequence[str],
    "Series",
    "np.ndarray[Any, Any]",
]
BooleanMask: TypeAlias = Union[
    Sequence[bool],
    "Series",
    "np.ndarray[Any, Any]",
]
SingleColSelector: TypeAlias = SingleIndexSelector | SingleNameSelector
MultiColSelector: TypeAlias = MultiIndexSelector | MultiNameSelector | BooleanMask

# LazyFrame engine selection
EngineType: TypeAlias = Union[
    Literal["auto", "in-memory", "streaming", "gpu"], "GPUEngine"
]

PlanStage: TypeAlias = Literal["ir", "physical"]

FileSource: TypeAlias = (
    str
    | Path
    | IO[bytes]
    | bytes
    | list[str]
    | list[Path]
    | list[IO[bytes]]
    | list[bytes]
)

JSONEncoder = Callable[[Any], bytes] | Callable[[Any], str]

DeprecationType: TypeAlias = Literal[
    "function",
    "renamed_parameter",
    "streaming_parameter",
    "nonkeyword_arguments",
    "parameter_as_multi_positional",
]


__all__ = [
    "Ambiguous",
    "ArrowArrayExportable",
    "ArrowStreamExportable",
    "AsofJoinStrategy",
    "AvroCompression",
    "BooleanMask",
    "BufferInfo",
    "CategoricalOrdering",
    "ClosedInterval",
    "ColumnFormatDict",
    "ColumnNameOrSelector",
    "ColumnTotalsDefinition",
    "ColumnWidthsDefinition",
    "ComparisonOperator",
    "ConcatMethod",
    "ConditionalFormatDict",
    "ConnectionOrCursor",
    "CorrelationMethod",
    "CsvEncoding",
    "CsvQuoteStyle",
    "Cursor",
    "DbReadEngine",
    "DbWriteEngine",
    "DbWriteMode",
    "DeprecationType",
    "Endianness",
    "EngineType",
    "EpochTimeUnit",
    "ExcelSpreadsheetEngine",
    "ExplainFormat",
    "FileSource",
    "FillNullStrategy",
    "FloatFmt",
    "FrameInitTypes",
    "FrameType",
    "IndexOrder",
    "InterpolationMethod",
    "IntoExpr",
    "IntoExprColumn",
    "IpcCompression",
    "JSONEncoder",
    "JaxExportType",
    "JoinStrategy",
    "JoinValidation",
    "Label",
    "ListToStructWidthStrategy",
    "MaintainOrderJoin",
    "MapElementsStrategy",
    "MultiColSelector",
    "MultiIndexSelector",
    "MultiNameSelector",
    "NdjsonCompression",
    "NonExistent",
    "NonNestedLiteral",
    "NullBehavior",
    "NumericLiteral",
    "OneOrMoreDataTypes",
    "Orientation",
    "ParallelStrategy",
    "ParametricProfileNames",
    "ParquetCompression",
    "PivotAgg",
    "PolarsDataType",
    "PolarsIntegerType",
    "PolarsTemporalType",
    "PolarsType",
    "PythonDataType",
    "PythonLiteral",
    "QuantileMethod",
    "RankMethod",
    "Roll",
    "RowTotalsDefinition",
    "SchemaDefinition",
    "SchemaDict",
    "SearchSortedSide",
    "SelectorType",
    "SerializationFormat",
    "SeriesBuffers",
    "SingleColSelector",
    "SingleIndexSelector",
    "SingleNameSelector",
    "SizeUnit",
    "StartBy",
    "SyncOnCloseMethod",
    "TemporalLiteral",
    "TimeUnit",
    "TorchExportType",
    "TransferEncoding",
    "UnicodeForm",
    "UniqueKeepStrategy",
    "UnstackDirection",
    "WindowMappingStrategy",
]


class ParquetMetadataContext:
    """
    The context given when writing file-level parquet metadata.

    .. warning::
        This functionality is considered **experimental**. It may be removed or
        changed at any point without it being considered a breaking change.
    """

    def __init__(self, *, arrow_schema: str) -> None:
        self.arrow_schema = arrow_schema

    arrow_schema: str  #: The base64 encoded arrow schema that is going to be written into metadata.


ParquetMetadataFn: TypeAlias = Callable[[ParquetMetadataContext], dict[str, str]]
ParquetMetadata: TypeAlias = dict[str, str] | ParquetMetadataFn

StorageOptionsDict: TypeAlias = dict[str, Any]
