from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict

from polars import col
from polars._utils.unstable import issue_unstable_warning
from polars.expr import Expr

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyExpr

    from pathlib import Path
    from typing import Callable

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyPartitioning


class PartitionKey(TypedDict):
    """
    A key-value pair that got used during paritioning.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Fields
    ------
    name
        Name of the key column.
    value
        Value of the key for this partition.
    """

    name: str
    value: str


class KeyedPartitionContext(TypedDict):
    """
    Callback context for a partition creation using keys.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Fields
    ------
    part
        The index of the created file starting from zero.
    keys
        All the key names and values used for this partition.
    file_path
        The chosen output path before the callback was called without `base_path`.
    full_path
        The chosen output path before the callback was called with `base_path`.
    """

    part: int
    keys: list[PartitionKey]
    file_path: Path
    full_path: Path


class BasePartitionContext(TypedDict):
    """
    Callback context for a partition creation.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Fields
    ------
    part
        The index of the created file starting from zero.
    file_path
        The chosen output path before the callback was called without `base_path`.
    full_path
        The chosen output path before the callback was called with `base_path`.
    """

    part: int
    file_path: Path
    full_path: Path


class PartitionMaxSize:
    """
    Partitioning scheme to write files with a maximum size.

    This partitioning scheme generates files that have a given maximum size. If
    the size reaches the maximum size, it is closed and a new file is opened.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Parameters
    ----------
    base_path
        The base path for the output files.
    file_path
        A callback to register or modify the output path for each partition
        offset by the `base_path`.
    max_size : int
        The maximum size in rows of each of the generated files.

    Examples
    --------
    Split a parquet file by over smaller CSV files with 100 000 rows each:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     PartitionMax("./out", max_size=100_000),
    ... )  # doctest: +SKIP
    """

    _p: PyPartitioning

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path: Callable[[BasePartitionContext], Path | str] | None = None,
        max_size: int,
    ) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")
        self._p = PyPartitioning.new_max_size(
            base_path=base_path, file_path_cb=file_path, max_size=max_size
        )

    @property
    def _base_path(self) -> str | None:
        return self._p.base_path


def _lower_by(
    by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
) -> list[PyExpr]:
    def to_expr(i: str | Expr) -> Expr:
        if isinstance(i, str):
            return col(i)
        else:
            return i

    lowered_by: list[PyExpr]
    if isinstance(by, str):
        lowered_by = [col(by)._pyexpr]
    elif isinstance(by, Expr):
        lowered_by = [by._pyexpr]
    elif isinstance(by, Sequence):
        lowered_by = [to_expr(e)._pyexpr for e in by]
    elif isinstance(by, Mapping):
        lowered_by = [e.alias(n)._pyexpr for n, e in by.items()]
    else:
        msg = "invalid `by` type"
        raise TypeError(msg)

    return lowered_by


class PartitionByKey:
    """
    Partitioning scheme to write files split by the values of keys.

    This partitioning scheme generates an arbitrary amount of files splitting
    the data depending on what the value is of key expressions.

    The amount of files that can be written is not limited. However, when
    writing beyond a certain amount of files, the data for the remaining
    partitions is buffered before writing to the file.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Parameters
    ----------
    base_path
        The base path for the output files.

        Use the `mkdir` option on the `sink_*` methods to ensure directories in
        the path are created.
    file_path
        A callback to register or modify the output path for each partition
        offset by the `base_path`.
    by
        The expressions to partition by.
    include_key : bool
        Whether to include the key columns in the output files.

    Examples
    --------
    Split into a hive-partitioning style partition:

    >>> (
    ...     pl.DataFrame({"a": [1, 2, 3], "b": [5, 7, 9], "c": ["A", "B", "C"]})
    ...     .lazy()
    ...     .sink_parquet(
    ...         PartitionByKey(
    ...             "./out",
    ...             by=[pl.col.a, pl.col.b],
    ...             include_key=False,
    ...         ),
    ...         mkdir=True,
    ...     )
    ... )  # doctest: +SKIP

    Split a parquet file by a column `year` into CSV files:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     PartitionByKey(
    ...         "./out/",
    ...         file_path=lambda ctx: f"year={ctx.keys[0].value}.csv",
    ...         by="year",
    ...     ),
    ... )  # doctest: +SKIP
    """

    _p: PyPartitioning

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path: Callable[[KeyedPartitionContext], Path | str] | None = None,
        by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
        include_key: bool = True,
    ) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")

        lowered_by = _lower_by(by)
        self._p = PyPartitioning.new_by_key(
            base_path=base_path,
            file_path_cb=file_path,
            by=lowered_by,
            include_key=include_key,
        )

    @property
    def _base_path(self) -> str | None:
        return self._p.base_path


class PartitionParted:
    """
    Partitioning scheme to split parted dataframes.

    This is a specialized version of `PartitionByKey`. Where as `PartitionByKey` accepts
    data in any order, this scheme expects the input data to be pre-grouped or
    pre-sorted. This scheme suffers a lot less overhead than `PartitionByKey`, but may
    not be always applicable.

    Each new value of the key expressions starts a new partition, therefore repeating
    the same value multiple times may overwrite previous partitions.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Parameters
    ----------
    base_path
        The base path for the output files.

        Use the `mkdir` option on the `sink_*` methods to ensure directories in
        the path are created.
    file_path
        A callback to register or modify the output path for each partition
        offset by the `base_path`.
    by
        The expressions to partition by.
    include_key : bool
        Whether to include the key columns in the output files.

    Examples
    --------
    Split a parquet file by a column `year` into CSV files:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     PartitionParted("./out", by="year"),
    ...     mkdir=True,
    ... )  # doctest: +SKIP
    """

    _p: PyPartitioning

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path: Callable[[KeyedPartitionContext], Path | str] | None = None,
        by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
        include_key: bool = True,
    ) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")

        lowered_by = _lower_by(by)
        self._p = PyPartitioning.new_by_key(
            base_path=base_path,
            file_path_cb=file_path,
            by=lowered_by,
            include_key=include_key,
        )

    @property
    def _base_path(self) -> str | None:
        return self._p.base_path
