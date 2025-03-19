from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from polars import col
from polars._utils.unstable import issue_unstable_warning
from polars.expr import Expr

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyExpr

    from pathlib import Path

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyPartitioning


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
    path
        The path to the output files. The format string `{part}` is replaced to the
        zero-based index of the file.
    max_size : int
        The maximum size in rows of each of the generated files.
    """

    _p: PyPartitioning

    def __init__(self, path: Path | str, *, max_size: int) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")
        self._p = PyPartitioning.new_max_size(path, max_size)

    @property
    def _path(self) -> str:
        return self._p.path


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
    path
        The format path to the output files. Format arguments:
        - `{part}` is replaced to the zero-based index of the file.
        - `{key[i].name}` is replaced by the name of key `i`.
        - `{key[i].value}` is replaced by the value of key `i`.

        Use the `mkdir` option on the `sink_*` methods to ensure directories in
        the path are created.
    by
        The expressions to partition by.
    include_key : bool
        Whether to include the key columns in the output files.

    Examples
    --------
    Split a parquet file by a column `year` into CSV files:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     PartitionByKey(
    ...         "./out/{key[0].value}.csv",
    ...         by="year",
    ...     ),
    ... )  # doctest: +SKIP

    Split into a hive-partitioning style partition:

    >>> (
    ...     pl.DataFrame({"a": [1, 2, 3], "b": [5, 7, 9], "c": ["A", "B", "C"]})
    ...     .lazy()
    ...     .sink_parquet(
    ...         PartitionByKey(
    ...             "{key[0].name}={key[0].value}/{key[1].name}={key[1].value}/000",
    ...             by=[pl.col.a, pl.col.b],
    ...             include_key=False,
    ...         ),
    ...         mkdir=True,
    ...     )
    ... )  # doctest: +SKIP
    """

    _p: PyPartitioning

    def __init__(
        self,
        path: Path | str,
        *,
        by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
        include_key: bool = True,
    ) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")

        lowered_by = _lower_by(by)
        self._p = PyPartitioning.new_by_key(
            path, by=lowered_by, include_key=include_key
        )

    @property
    def _path(self) -> str:
        return self._p.path


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
    path
        The format path to the output files. Format arguments:
        - `{part}` is replaced to the zero-based index of the file.
        - `{key[i].name}` is replaced by the name of key `i`.
        - `{key[i].value}` is replaced by the value of key `i`.

        Use the `mkdir` option on the `sink_*` methods to ensure directories in
        the path are created.
    by
        The expressions to partition by.
    include_key : bool
        Whether to include the key columns in the output files.

    Examples
    --------
    Split a parquet file by a column `year` into CSV files:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     PartitionParted(
    ...         "./out/{key[0].value}.csv",
    ...         by="year",
    ...     ),
    ... )  # doctest: +SKIP
    """

    _p: PyPartitioning

    def __init__(
        self,
        path: Path | str,
        *,
        by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
        include_key: bool = True,
    ) -> None:
        issue_unstable_warning("Partitioning strategies are considered unstable.")

        lowered_by = _lower_by(by)
        self._p = PyPartitioning.new_by_key(
            path, by=lowered_by, include_key=include_key
        )

    @property
    def _path(self) -> str:
        return self._p.path
