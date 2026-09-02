"""Build lazy sink plans.

Attaching a sink node to a query plan is engine-independent: the engine is only
needed when the plan is executed. These helpers are shared by
`LazyFrame.sink_*`, which may return such a plan unexecuted (`lazy=True`), and
by the local engines, which execute it immediately.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from polars._utils.unstable import issue_unstable_warning
from polars._utils.various import normalize_filepath, qualified_type_name
from polars._utils.wrap import wrap_df, wrap_ldf

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO

    from polars._typing import (
        ArrowSchemaExportable,
        CsvQuoteStyle,
        IpcCompression,
        ParquetCompression,
        ParquetMetadata,
        StorageOptionsDict,
        SyncOnCloseMethod,
    )
    from polars.dataframe import DataFrame
    from polars.interchange.protocol import CompatLevel
    from polars.io.cloud import CredentialProviderFunction
    from polars.io.partition import PartitionBy, SinkedPathsCallback
    from polars.lazyframe.frame import LazyFrame


def _to_sink_target(
    path: str | Path | IO[bytes] | IO[str] | PartitionBy,
) -> str | Path | IO[bytes] | IO[str] | PartitionBy:
    from polars.io.partition import PartitionBy

    if isinstance(path, (str, Path)):
        return normalize_filepath(path)
    elif isinstance(path, io.IOBase):
        return path
    elif isinstance(path, PartitionBy):
        return path
    elif callable(getattr(path, "write", None)):
        # This allows for custom writers
        return path
    else:
        msg = f"`path` argument has invalid type {qualified_type_name(path)!r}, and cannot be turned into a sink target"
        raise TypeError(msg)


def _sink_parquet_plan(
    lf: LazyFrame,
    path: str | Path | IO[bytes] | PartitionBy,
    *,
    compression: ParquetCompression,
    compression_level: int | None,
    statistics: bool | str | dict[str, bool],
    row_group_size: int | None,
    data_page_size: int | None,
    maintain_order: bool,
    storage_options: StorageOptionsDict | None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None,
    sync_on_close: SyncOnCloseMethod | None,
    metadata: ParquetMetadata | None,
    arrow_schema: ArrowSchemaExportable | None,
    mkdir: bool,
    sinked_paths_callback: SinkedPathsCallback | None,
) -> LazyFrame:
    """Attach a parquet sink node to the plan, without executing it."""
    from polars._utils.parquet import wrap_parquet_metadata_callback
    from polars.io.cloud.credential_provider._builder import (
        _init_credential_provider_builder,
    )
    from polars.io.partition import _SinkOptions

    if metadata is not None:
        msg = "`metadata` parameter is considered experimental"
        issue_unstable_warning(msg)

    if arrow_schema is not None:
        msg = "`arrow_schema` parameter is considered unstable"
        issue_unstable_warning(msg)

    if isinstance(statistics, bool) and statistics:
        statistics = {
            "min": True,
            "max": True,
            "distinct_count": False,
            "null_count": True,
        }
    elif isinstance(statistics, bool) and not statistics:
        statistics = {}
    elif statistics == "full":
        statistics = {
            "min": True,
            "max": True,
            "distinct_count": True,
            "null_count": True,
        }

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, path, storage_options, "sink_parquet"
    )

    target = _to_sink_target(path)

    if isinstance(metadata, dict):
        if metadata:
            metadata = list(metadata.items())  # type: ignore[assignment]
        else:
            # Handle empty dict input
            metadata = None
    elif callable(metadata):
        metadata = wrap_parquet_metadata_callback(metadata)  # type: ignore[assignment]

    sink_options = _SinkOptions(
        mkdir=mkdir,
        maintain_order=maintain_order,
        sync_on_close=sync_on_close,
        storage_options=storage_options,
        credential_provider=credential_provider_builder,
        sinked_paths_callback=sinked_paths_callback,
    )

    ldf_py = lf._ldf.sink_parquet(
        target=target,
        sink_options=sink_options,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
        row_group_size=row_group_size,
        data_page_size=data_page_size,
        metadata=metadata,
        arrow_schema=arrow_schema,
    )
    return wrap_ldf(ldf_py)


def _sink_ipc_plan(
    lf: LazyFrame,
    path: str | Path | IO[bytes] | PartitionBy,
    *,
    compression: IpcCompression | None,
    compat_level: CompatLevel | None,
    record_batch_size: int | None,
    maintain_order: bool,
    storage_options: StorageOptionsDict | None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None,
    sync_on_close: SyncOnCloseMethod | None,
    mkdir: bool,
    _record_batch_statistics: bool,
    sinked_paths_callback: SinkedPathsCallback | None,
) -> LazyFrame:
    """Attach an IPC sink node to the plan, without executing it."""
    from polars.interchange.protocol import CompatLevel
    from polars.io.cloud.credential_provider._builder import (
        _init_credential_provider_builder,
    )
    from polars.io.partition import _SinkOptions

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, path, storage_options, "sink_ipc"
    )

    target = _to_sink_target(path)

    compat_level_py: int | bool
    if compat_level is None:
        compat_level_py = True
    elif isinstance(compat_level, CompatLevel):
        compat_level_py = compat_level._version
    else:
        msg = f"`compat_level` has invalid type: {qualified_type_name(compat_level)!r}"
        raise TypeError(msg)

    if compression is None:
        compression = "uncompressed"

    sink_options = _SinkOptions(
        mkdir=mkdir,
        maintain_order=maintain_order,
        sync_on_close=sync_on_close,
        storage_options=storage_options,
        credential_provider=credential_provider_builder,
        sinked_paths_callback=sinked_paths_callback,
    )

    ldf_py = lf._ldf.sink_ipc(
        target=target,
        sink_options=sink_options,
        compression=compression,
        compat_level=compat_level_py,
        record_batch_size=record_batch_size,
        record_batch_statistics=_record_batch_statistics,
    )
    return wrap_ldf(ldf_py)


def _sink_csv_plan(
    lf: LazyFrame,
    path: str | Path | IO[bytes] | IO[str] | PartitionBy,
    *,
    include_bom: bool,
    compression: Literal["uncompressed", "gzip", "zstd"],
    compression_level: int | None,
    check_extension: bool,
    include_header: bool,
    separator: str,
    line_terminator: str,
    quote_char: str,
    batch_size: int,
    datetime_format: str | None,
    date_format: str | None,
    time_format: str | None,
    float_scientific: bool | None,
    float_precision: int | None,
    decimal_comma: bool,
    null_value: str | None,
    quote_style: CsvQuoteStyle | None,
    maintain_order: bool,
    storage_options: StorageOptionsDict | None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None,
    sync_on_close: SyncOnCloseMethod | None,
    mkdir: bool,
) -> LazyFrame:
    """Attach a CSV sink node to the plan, without executing it."""
    from polars.io.cloud.credential_provider._builder import (
        _init_credential_provider_builder,
    )
    from polars.io.csv._utils import _check_arg_is_1byte
    from polars.io.partition import _SinkOptions

    _check_arg_is_1byte("separator", separator, can_be_empty=False)
    _check_arg_is_1byte("quote_char", quote_char, can_be_empty=False)
    if not null_value:
        null_value = None

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, path, storage_options, "sink_csv"
    )

    target = _to_sink_target(path)

    sink_options = _SinkOptions(
        mkdir=mkdir,
        maintain_order=maintain_order,
        sync_on_close=sync_on_close,
        storage_options=storage_options,
        credential_provider=credential_provider_builder,
    )

    ldf_py = lf._ldf.sink_csv(
        target=target,
        sink_options=sink_options,
        include_bom=include_bom,
        compression=compression,
        compression_level=compression_level,
        check_extension=check_extension,
        include_header=include_header,
        separator=ord(separator),
        line_terminator=line_terminator,
        quote_char=ord(quote_char),
        batch_size=batch_size,
        datetime_format=datetime_format,
        date_format=date_format,
        time_format=time_format,
        float_scientific=float_scientific,
        float_precision=float_precision,
        decimal_comma=decimal_comma,
        null_value=null_value,
        quote_style=quote_style,
    )
    return wrap_ldf(ldf_py)


def _sink_ndjson_plan(
    lf: LazyFrame,
    path: str | Path | IO[bytes] | IO[str] | PartitionBy,
    *,
    compression: Literal["uncompressed", "gzip", "zstd"],
    compression_level: int | None,
    check_extension: bool,
    maintain_order: bool,
    storage_options: StorageOptionsDict | None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None,
    sync_on_close: SyncOnCloseMethod | None,
    mkdir: bool,
) -> LazyFrame:
    """Attach an NDJSON sink node to the plan, without executing it."""
    from polars.io.cloud.credential_provider._builder import (
        _init_credential_provider_builder,
    )
    from polars.io.partition import _SinkOptions

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, path, storage_options, "sink_ndjson"
    )

    target = _to_sink_target(path)

    sink_options = _SinkOptions(
        mkdir=mkdir,
        maintain_order=maintain_order,
        sync_on_close=sync_on_close,
        storage_options=storage_options,
        credential_provider=credential_provider_builder,
    )

    ldf_py = lf._ldf.sink_ndjson(
        target=target,
        compression=compression,
        compression_level=compression_level,
        check_extension=check_extension,
        sink_options=sink_options,
    )
    return wrap_ldf(ldf_py)


def _sink_batches_plan(
    lf: LazyFrame,
    function: Callable[[DataFrame], bool | None],
    *,
    chunk_size: int | None,
    maintain_order: bool,
) -> LazyFrame:
    """Attach a batch callback sink node to the plan, without executing it."""

    def _wrap(pydf: Any) -> bool:
        return bool(function(wrap_df(pydf)))

    ldf_py = lf._ldf.sink_batches(
        function=_wrap,
        maintain_order=maintain_order,
        chunk_size=chunk_size,
    )
    return wrap_ldf(ldf_py)
