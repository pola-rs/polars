from __future__ import annotations

import ast
from _ast import GtE, Lt, LtE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiceberg.table import Table

    from polars import DataFrame, LazyFrame, Series

from ast import (
    Attribute,
    BinOp,
    BitAnd,
    BitOr,
    Call,
    Compare,
    Constant,
    Eq,
    Gt,
    Invert,
    List,
    UnaryOp,
)
from functools import partial, singledispatch
from typing import Any

import polars._reexport as pl
from polars.dependencies import _PYICEBERG_AVAILABLE

if _PYICEBERG_AVAILABLE:
    from pyiceberg.expressions import (
        And,
        EqualTo,
        GreaterThan,
        GreaterThanOrEqual,
        In,
        IsNaN,
        IsNull,
        LessThan,
        LessThanOrEqual,
        Not,
        Or,
    )


def scan_iceberg(
    source: str | Table,
    *,
    storage_options: dict[str, Any] | None = None,
) -> LazyFrame:
    """
    Lazily read from an Apache Iceberg table.

    Parameters
    ----------
    source
        URI or Table to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported but
        for the supported object storages - GCS, Azure and S3 full URI must be provided.
    storage_options
        Extra options for the storage backends supported by `pyiceberg`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here <https://py.iceberg.apache.org/configuration/>`__.

    Returns
    -------
    LazyFrame

    Examples
    --------
    Creates a scan for a Iceberg table from local filesystem, or object store.

    >>> table_path = "file:/path/to/iceberg-table/metadata.json"
    >>> pl.scan_iceberg(table_path).collect()  # doctest: +SKIP

    Creates a scan for an Iceberg table from S3.
    See a list of supported storage options for S3 `here
    <https://py.iceberg.apache.org/configuration/#fileio>`__.

    >>> table_path = "s3://bucket/path/to/iceberg-table/metadata.json"
    >>> storage_options = {
    ...     "s3.region": "eu-central-1",
    ...     "s3.access-key-id": "THE_AWS_ACCESS_KEY_ID",
    ...     "s3.secret-access-key": "THE_AWS_SECRET_ACCESS_KEY",
    ... }
    >>> pl.scan_iceberg(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table from Azure.
    Supported options for Azure are available `here
    <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants>`__.

    Following type of table paths are supported,
    * az://<container>/<path>/metadata.json
    * adl://<container>/<path>/metadata.json
    * abfs[s]://<container>/<path>/metadata.json

    >>> table_path = "az://container/path/to/iceberg-table/metadata.json"
    >>> storage_options = {
    ...     "adlfs.account-name": "AZURE_STORAGE_ACCOUNT_NAME",
    ...     "adlfs.account-key": "AZURE_STORAGE_ACCOUNT_KEY",
    ... }
    >>> pl.scan_iceberg(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table with additional delta specific options.
    In the below example, `without_files` option is used which loads the table without
    file tracking information.

    >>> table_path = "/path/to/iceberg-table/metadata.json"
    >>> delta_table_options = {"without_files": True}
    >>> pl.scan_iceberg(
    ...     table_path, delta_table_options=delta_table_options
    ... ).collect()  # doctest: +SKIP

    """
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.table import StaticTable

    if isinstance(source, str):
        source = StaticTable.from_metadata(
            metadata_location=source, properties=storage_options or {}
        )

    func = partial(_scan_pyarrow_dataset_impl, source)
    arrow_schema = schema_to_pyarrow(source.schema())
    return pl.LazyFrame._scan_python_function(arrow_schema, func, pyarrow=True)


def _to_ast(expr: str) -> Any:
    return ast.parse(expr, mode="eval").body


def _scan_pyarrow_dataset_impl(
    tbl: Table,
    with_columns: list[str] | None = None,
    predicate: str = "",
    n_rows: int | None = None,
    **kwargs: Any,
) -> DataFrame | Series:
    """
    Take the projected columns and materialize an arrow table.

    Parameters
    ----------
    tbl
        pyarrow dataset
    with_columns
        Columns that are projected
    predicate
        pyarrow expression that can be evaluated with eval
    n_rows:
        Materialize only n rows from the arrow dataset
    batch_size
        The maximum row count for scanned pyarrow record batches.
    kwargs:
        For backward compatibility

    Returns
    -------
    DataFrame

    """
    scan = tbl.scan(limit=n_rows)

    if with_columns is not None:
        scan = scan.select(*with_columns)

    if predicate is not None:
        try:
            expr_ast = _to_ast(predicate)
            pyiceberg_expr = _convert_predicate(expr_ast)
        except ValueError as e:
            raise ValueError(
                f"Could not convert predicate to PyIceberg: {predicate}"
            ) from e

        scan = scan.filter(pyiceberg_expr)

    from polars import from_arrow

    return from_arrow(scan.to_arrow())


@singledispatch
def _convert_predicate(a: Any) -> Any:
    """Walks the AST to  convert the  PyArrow expression to a PyIceberg expression."""
    raise ValueError(f"Unexpected symbol: {a}")


@_convert_predicate.register(Constant)
def _(a: Constant) -> Any:
    return a.value


@_convert_predicate.register(UnaryOp)
def _(a: UnaryOp) -> Any:
    if isinstance(a.op, Invert):
        return Not(_convert_predicate(a.operand))
    else:
        raise ValueError(f"Unexpected UnaryOp: {a}")


@_convert_predicate.register(Call)
def _(a: Call) -> Any:
    args = [_convert_predicate(arg) for arg in a.args]
    f = _convert_predicate(a.func)
    if f == "field":
        return args
    else:
        ref = _convert_predicate(a.func.value)[0]  # type: ignore[attr-defined]
        if f == "isin":
            return In(ref, args[0])
        elif f == "is_null":
            return IsNull(ref)
        elif f == "is_nan":
            return IsNaN(ref)

    raise ValueError(f"Unknown call: {f}")


@_convert_predicate.register(Attribute)
def _(a: Attribute) -> Any:
    return a.attr


@_convert_predicate.register(BinOp)
def _(a: BinOp) -> Any:
    lhs = _convert_predicate(a.left)
    rhs = _convert_predicate(a.right)

    op = a.op
    if isinstance(op, BitAnd):
        return And(lhs, rhs)
    if isinstance(op, BitOr):
        return Or(lhs, rhs)
    else:
        raise ValueError(f"Unknown: {lhs} {op} {rhs}")


@_convert_predicate.register(Compare)
def _(a: Compare) -> Any:
    op = a.ops[0]
    lhs = _convert_predicate(a.left)[0]
    rhs = _convert_predicate(a.comparators[0])

    if isinstance(op, Gt):
        return GreaterThan(lhs, rhs)
    if isinstance(op, GtE):
        return GreaterThanOrEqual(lhs, rhs)
    if isinstance(op, Eq):
        return EqualTo(lhs, rhs)
    if isinstance(op, Lt):
        return LessThan(lhs, rhs)
    if isinstance(op, LtE):
        return LessThanOrEqual(lhs, rhs)
    else:
        raise ValueError(f"Unknown comparison: {op}")


@_convert_predicate.register(List)
def _(a: List) -> Any:
    return [_convert_predicate(e) for e in a.elts]


def _check_if_pyiceberg_available() -> None:
    if not _PYICEBERG_AVAILABLE:
        raise ImportError(
            "pyiceberg is not installed. Please run `pip install pyiceberg[pyarrow]>=0.4.0`."
        )
