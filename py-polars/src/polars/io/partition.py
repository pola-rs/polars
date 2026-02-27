from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from polars._utils.parse.expr import parse_into_list_of_expressions
from polars._utils.unstable import issue_unstable_warning

if TYPE_CHECKING:
    import contextlib
    from pathlib import Path

    from polars import DataFrame

    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars._plr import PyExpr

    from collections.abc import Callable, Sequence
    from typing import IO

    from polars._typing import StorageOptionsDict, SyncOnCloseMethod
    from polars.expr import Expr
    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder


class PartitionBy:
    """
    Configuration for writing to multiple output files.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.

    Parameters
    ----------
    base_path
        Base path to write to.
    file_path_provider
        Callable for custom file output paths.
    key
        Expressions to partition by.
    include_key
        Include the partition key expression outputs in the output files.
    max_rows_per_file
        Maximum number of rows to write for each file. Note that files may have
        less than this amount of rows.
    approximate_bytes_per_file
        Approximate number of bytes to write to each file. This is measured as
        the estimated size of the DataFrame in memory.

    Examples
    --------
    Split to multiple files partitioned by year:

    >>> pl.LazyFrame({"year": [2026, 2027, 1970], "month": [0, 0, 0]}).sink_parquet(
    ...     pl.PartitionBy("data/", key="year")
    ... )  # doctest: +SKIP

    Split to multiple files based on size:

    >>> pl.LazyFrame({"year": [2026, 2027, 1970], "month": [0, 0, 0]}).sink_parquet(
    ...     pl.PartitionBy(
    ...         "data/", max_rows_per_file=1000, approximate_bytes_per_file=100_000_000
    ...     )
    ... )  # doctest: +SKIP

    Split to multiple files partitioned by year, with limits on individual file sizes:

    >>> pl.LazyFrame({"year": [2026, 2027, 1970], "month": [0, 0, 0]}).sink_parquet(
    ...     pl.PartitionBy(
    ...         "data/",
    ...         key="year",
    ...         max_rows_per_file=1000,
    ...         approximate_bytes_per_file=100_000_000,
    ...     )
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        base_path: str | Path,
        *,
        file_path_provider: Callable[
            [FileProviderArgs], str | Path | IO[bytes] | IO[str]
        ]
        | None = None,
        key: str | Expr | Sequence[str | Expr] | Mapping[str, Expr] | None = None,
        include_key: bool | None = None,
        max_rows_per_file: int | None = None,
        approximate_bytes_per_file: int | Literal["auto"] | None = "auto",
    ) -> None:
        msg = "`PartitionBy` functionality is considered unstable"
        issue_unstable_warning(msg)

        if (
            key is None
            and max_rows_per_file is None
            and approximate_bytes_per_file == "auto"
        ):
            msg = (
                "at least one of "
                "('key', 'max_rows_per_file', 'approximate_bytes_per_file') "
                "must be specified for PartitionBy"
            )
            raise ValueError(msg)

        if key is None and include_key is not None:
            msg = "cannot use 'include_key' without specifying 'key'"
            raise ValueError(msg)

        base_path = str(base_path)

        if approximate_bytes_per_file == "auto":
            approximate_bytes_per_file = (
                4_294_967_295 if max_rows_per_file is None else None
            )

        if approximate_bytes_per_file is None:
            approximate_bytes_per_file = (1 << 64) - 1

        self._pl_partition_by = _PartitionByInner(
            base_path=base_path,
            file_path_provider=file_path_provider,
            key=_parse_to_pyexpr_list(key) if key is not None else None,
            include_key=include_key,
            max_rows_per_file=max_rows_per_file,
            approximate_bytes_per_file=approximate_bytes_per_file,
        )


@dataclass(kw_only=True)
class FileProviderArgs:
    """
    Holds information on the file being sinked to.

    .. warning::
        This functionality is currently considered **unstable**. It may be
        changed at any point without it being considered a breaking change.
    """

    index_in_partition: int
    partition_keys: DataFrame


@dataclass(kw_only=True)
class _PartitionByInner:
    """
    Holds parsed partitioned sink options.

    For internal use.
    """

    base_path: str
    file_path_provider: (
        Callable[[FileProviderArgs], str | Path | IO[bytes] | IO[str]] | None
    )
    key: list[PyExpr] | None
    include_key: bool | None
    max_rows_per_file: int | None
    approximate_bytes_per_file: int


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
    storage_options: StorageOptionsDict | None = None
    credential_provider: CredentialProviderBuilder | None = None


def _parse_to_pyexpr_list(
    exprs_or_columns: str | Expr | Sequence[str | Expr] | Mapping[str, Expr],
) -> list[PyExpr]:
    if isinstance(exprs_or_columns, Mapping):
        return [e.alias(k)._pyexpr for k, e in exprs_or_columns.items()]

    return parse_into_list_of_expressions(exprs_or_columns)
