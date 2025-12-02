from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polars import DataFrame
from polars._utils.parse.expr import parse_into_list_of_expressions
from polars._utils.unstable import issue_unstable_warning

if TYPE_CHECKING:
    import contextlib

    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars._plr import PyDataFrame, PyExpr

    from collections.abc import Sequence
    from typing import IO, Any, Callable

    from polars._typing import SyncOnCloseMethod
    from polars.expr import Expr
    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder


class KeyedPartition:
    """
    A key-value pair for a partition.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    See Also
    --------
    PartitionByKey
    PartitionParted
    KeyedPartitionContext
    """

    def __init__(self, name: str, str_value: str, raw_value: Any) -> None:
        self.name = name
        self.str_value = str_value
        self.raw_value = raw_value

    name: str  #: Name of the key column.
    str_value: str  #: Value of the key as a path and URL safe string.
    raw_value: Any  #: Value of the key for this partition.

    def hive_name(self) -> str:
        """Get the `key=value`."""
        return f"{self.name}={self.str_value}"


class KeyedPartitionContext:
    """
    Callback context for a partition creation using keys.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    See Also
    --------
    PartitionByKey
    PartitionParted
    """

    def __init__(
        self,
        file_idx: int,
        part_idx: int,
        in_part_idx: int,
        keys: list[KeyedPartition],
        file_path: Path,
        full_path: Path,
    ) -> None:
        self.file_idx = file_idx
        self.part_idx = part_idx
        self.in_part_idx = in_part_idx
        self.keys = keys
        self.file_path = file_path
        self.full_path = full_path

    file_idx: int  #: The index of the created file starting from zero.
    part_idx: int  #: The index of the created partition starting from zero.
    in_part_idx: int  #: The index of the file within this partition starting from zero.
    keys: list[KeyedPartition]  #: All the key names and values used for this partition.
    file_path: Path  #: The chosen output path before the callback was called without `base_path`.
    full_path: (
        Path  #: The chosen output path before the callback was called with `base_path`.
    )

    def hive_dirs(self) -> Path:
        """The keys mapped to hive directories."""
        assert len(self.keys) > 0
        p = Path(self.keys[0].hive_name())
        for key in self.keys[1:]:
            p /= Path(key.hive_name())
        return p


class BasePartitionContext:
    """
    Callback context for a partition creation.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    See Also
    --------
    PartitionMaxSize
    """

    def __init__(self, file_idx: int, file_path: Path, full_path: Path) -> None:
        self.file_idx = file_idx
        self.file_path = file_path
        self.full_path = full_path

    file_idx: int  #: The index of the created file starting from zero.
    file_path: Path  #: The chosen output path before the callback was called without `base_path`.
    full_path: (
        Path  #: The chosen output path before the callback was called with `base_path`.
    )


# TODO: Expose this as Python API (as unstable).
class _SinkDirectory:
    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path_provider: Callable[
            [KeyedPartitionContext], Path | str | IO[bytes] | IO[str]
        ]
        | None = None,
        partition_by: str
        | Expr
        | Sequence[str | Expr]
        | Mapping[str, Expr]
        | None = None,
        partition_keys_sorted: bool | None = None,
        include_keys: bool | None = None,
        per_partition_sort_by: str | Expr | Sequence[str | Expr] | None = None,
        per_file_sort_by: str | Expr | Sequence[str | Expr] | None = None,
        max_rows_per_file: int | None = None,
        finish_callback: Callable[[DataFrame], None] | None = None,
    ) -> None:
        base_path = str(base_path)

        self._pl_sink_directory = _SinkDirectoryInner(
            base_path=base_path,
            file_path_provider=file_path_provider,
            partition_by=(
                _parse_to_pyexpr_list(partition_by)
                if partition_by is not None
                else None
            ),
            partition_keys_sorted=partition_keys_sorted,
            include_keys=include_keys,
            per_partition_sort_by=(
                _parse_to_pyexpr_list(per_partition_sort_by)
                if per_partition_sort_by is not None
                else None
            ),
            per_file_sort_by=(
                _parse_to_pyexpr_list(per_file_sort_by)
                if per_file_sort_by is not None
                else None
            ),
            max_rows_per_file=max_rows_per_file,
            finish_callback=(
                _prepare_finish_callback(finish_callback)
                if finish_callback is not None
                else None
            ),
        )

    @property
    def _base_path(self) -> str | None:
        return self._pl_sink_directory.base_path


def _parse_to_pyexpr_list(
    exprs_or_columns: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
) -> list[PyExpr]:
    if isinstance(exprs_or_columns, Mapping):
        return [e.alias(k)._pyexpr for k, e in exprs_or_columns.items()]

    return parse_into_list_of_expressions(exprs_or_columns)


def _cast_base_file_path_cb(
    file_path_cb: Callable[[BasePartitionContext], Path | str | IO[bytes] | IO[str]]
    | None,
) -> Callable[[KeyedPartitionContext], Path | str | IO[bytes] | IO[str]] | None:
    if file_path_cb is None:
        return None
    return lambda ctx: file_path_cb(
        BasePartitionContext(
            file_idx=ctx.file_idx,
            file_path=Path(ctx.file_path),
            full_path=Path(ctx.full_path),
        )
    )


def _cast_keyed_file_path_cb(
    file_path_cb: Callable[[KeyedPartitionContext], Path | str | IO[bytes] | IO[str]]
    | None,
) -> Callable[[KeyedPartitionContext], Path | str | IO[bytes] | IO[str]] | None:
    if file_path_cb is None:
        return None
    return lambda ctx: file_path_cb(
        KeyedPartitionContext(
            file_idx=ctx.file_idx,
            part_idx=ctx.part_idx,
            in_part_idx=ctx.in_part_idx,
            keys=[
                KeyedPartition(
                    name=kv.name, str_value=kv.str_value, raw_value=kv.raw_value
                )
                for kv in ctx.keys
            ],
            file_path=Path(ctx.file_path),
            full_path=Path(ctx.full_path),
        )
    )


def _prepare_finish_callback(
    f: Callable[[DataFrame], None] | None,
) -> Callable[[PyDataFrame], None] | None:
    if f is None:
        return None

    def cb(pydf: PyDataFrame) -> None:
        nonlocal f
        f(DataFrame._from_pydf(pydf))

    return cb


class PartitionMaxSize(_SinkDirectory):
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
        relative to the `base_path`. The callback provides a
        :class:`polars.io.partition.BasePartitionContext` that contains information
        about the partition.

        If no callback is given, it defaults to `{ctx.file_idx}.{EXT}`.
    max_size : int
        The maximum size in rows of each of the generated files.
    per_partition_sort_by
        Columns or expressions to sort over within each partition.

        Note that this might increase the memory consumption needed for each partition.
    finish_callback
        A callback that gets called when the query finishes successfully.

        For parquet files, the callback is given a dataframe with metrics about all
        files written files.

    Examples
    --------
    Split a parquet file by over smaller CSV files with 100 000 rows each:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     pl.PartitionMax("./out/", max_size=100_000),
    ... )  # doctest: +SKIP

    See Also
    --------
    PartitionByKey
    PartitionParted
    polars.io.partition.BasePartitionContext
    """

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path: Callable[[BasePartitionContext], Path | str | IO[bytes] | IO[str]]
        | None = None,
        max_size: int,
        finish_callback: Callable[[DataFrame], None] | None = None,
    ) -> None:
        issue_unstable_warning("partitioning strategies are considered unstable.")

        file_path_provider = _cast_base_file_path_cb(file_path)

        super().__init__(
            base_path=base_path,
            file_path_provider=file_path_provider,
            max_rows_per_file=max_size,
            finish_callback=finish_callback,
        )


class PartitionByKey(_SinkDirectory):
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
        relative to the `base_path`. The callback provides a
        :class:`polars.io.partition.KeyedPartitionContext` that contains information
        about the partition.

        If no callback is given, it defaults to
        `{ctx.keys.hive_dirs()}/{ctx.in_part_idx}.{EXT}`.
    by
        The expressions to partition by.
    include_key : bool
        Whether to include the key columns in the output files.
    per_partition_sort_by
        Columns or expressions to sort over within each partition.

        Note that this might increase the memory consumption needed for each partition.
    finish_callback
        A callback that gets called when the query finishes successfully.

        For parquet files, the callback is given a dataframe with metrics about all
        files written files.

    Examples
    --------
    Split into a hive-partitioning style partition:

    >>> (
    ...     pl.LazyFrame(
    ...         {"a": [1, 2, 3], "b": [5, 7, 9], "c": ["A", "B", "C"]}
    ...     ).sink_parquet(
    ...         pl.PartitionByKey(
    ...             "./out/",
    ...             by=["a", "b"],
    ...             include_key=False,
    ...         ),
    ...         mkdir=True,
    ...     )
    ... )  # doctest: +SKIP

    Split a parquet file by a column `year` into CSV files:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     PartitionByKey(
    ...         "./out/",
    ...         file_path=lambda ctx: f"year={ctx.keys[0].str_value}.csv",
    ...         by="year",
    ...     ),
    ... )  # doctest: +SKIP

    See Also
    --------
    PartitionMaxSize
    PartitionParted
    polars.io.partition.KeyedPartitionContext
    """

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path: Callable[[KeyedPartitionContext], Path | str | IO[bytes] | IO[str]]
        | None = None,
        by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
        include_key: bool = True,
        per_partition_sort_by: str | Expr | Sequence[str | Expr] | None = None,
        finish_callback: Callable[[DataFrame], None] | None = None,
    ) -> None:
        issue_unstable_warning("partitioning strategies are considered unstable.")

        super().__init__(
            base_path=base_path,
            file_path_provider=_cast_keyed_file_path_cb(file_path),
            partition_by=by,
            include_keys=include_key,
            per_partition_sort_by=per_partition_sort_by,
            finish_callback=finish_callback,
        )


class PartitionParted(_SinkDirectory):
    """
    Partitioning scheme to split parted dataframes.

    This is a specialized version of :class:`PartitionByKey`. Where as
    :class:`PartitionByKey` accepts data in any order, this scheme expects the input
    data to be pre-grouped or pre-sorted. This scheme suffers a lot less overhead than
    :class:`PartitionByKey`, but may not be always applicable.

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
        relative to the `base_path`.The callback provides a
        :class:`polars.io.partition.KeyedPartitionContext` that contains information
        about the partition.

        If no callback is given, it defaults to
        `{ctx.keys.hive_dirs()}/{ctx.in_part_idx}.{EXT}`.
    by
        The expressions to partition by.
    include_key : bool
        Whether to include the key columns in the output files.
    per_partition_sort_by
        Columns or expressions to sort over within each partition.

        Note that this might increase the memory consumption needed for each partition.
    finish_callback
        A callback that gets called when the query finishes successfully.

        For parquet files, the callback is given a dataframe with metrics about all
        files written files.

    Examples
    --------
    Split a parquet file by a column `year` into CSV files:

    >>> pl.scan_parquet("/path/to/file.parquet").sink_csv(
    ...     pl.PartitionParted("./out/", by="year"),
    ...     mkdir=True,
    ... )  # doctest: +SKIP

    See Also
    --------
    PartitionMaxSize
    PartitionByKey
    polars.io.partition.KeyedPartitionContext
    """

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path: Callable[[KeyedPartitionContext], Path | str | IO[bytes] | IO[str]]
        | None = None,
        by: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
        include_key: bool = True,
        per_partition_sort_by: str | Expr | Sequence[str | Expr] | None = None,
        finish_callback: Callable[[DataFrame], None] | None = None,
    ) -> None:
        issue_unstable_warning("partitioning strategies are considered unstable.")

        super().__init__(
            base_path=base_path,
            file_path_provider=_cast_keyed_file_path_cb(file_path),
            partition_by=by,
            partition_keys_sorted=True,
            include_keys=include_key,
            per_partition_sort_by=per_partition_sort_by,
            finish_callback=finish_callback,
        )


# TODO: Add `kw_only=True` after 3.9 support dropped
@dataclass
class _SinkDirectoryInner:
    """
    Holds parsed directory sink options.

    For internal use.
    """

    base_path: str
    file_path_provider: (
        Callable[[KeyedPartitionContext], Path | str | IO[bytes] | IO[str]] | None
    )
    partition_by: list[PyExpr] | None
    partition_keys_sorted: bool | None
    include_keys: bool | None
    per_partition_sort_by: list[PyExpr] | None
    per_file_sort_by: list[PyExpr] | None
    max_rows_per_file: int | None
    finish_callback: Callable[[PyDataFrame], None] | None


@dataclass
class _SinkOptions:
    """
    Holds sink options that are generic over file / target type.

    For internal use. Most of the options will parse into `UnifiedSinkArgs`.
    """

    mkdir: bool
    maintain_order: bool
    sync_on_close: SyncOnCloseMethod | None = None

    # Cloud
    storage_options: list[tuple[str, str]] | None = None
    credential_provider: CredentialProviderBuilder | None = None
    retries: int = 2
