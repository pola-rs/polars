"""Query execution engines."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, ClassVar, Literal, overload

from polars._dependencies import import_optional
from polars._utils.async_ import _AioDataFrameResult, _GeventDataFrameResult
from polars._utils.deprecation import issue_deprecation_warning
from polars._utils.unstable import issue_unstable_warning
from polars._utils.various import normalize_filepath, qualified_type_name
from polars._utils.wrap import wrap_df, wrap_ldf
from polars._warnings import issue_warning
from polars.lazyframe.in_process import InProcessQuery
from polars.lazyframe.query_result import SingleNodeQueryResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from rmm.mr import DeviceMemoryResource  # type: ignore[import-not-found]

    from polars._plr import PyCollectBatches, PyLazyFrame
    from polars._typing import (
        ArrowSchemaExportable,
        AsyncResult,
        CsvQuoteStyle,
        IpcCompression,
        ParquetCompression,
        ParquetMetadata,
        PostOptCallback,
        StorageOptionsDict,
        SyncOnCloseMethod,
    )
    from polars.dataframe import DataFrame
    from polars.interchange.protocol import CompatLevel
    from polars.io.cloud import CredentialProviderFunction
    from polars.io.partition import PartitionBy, SinkedPathsCallback
    from polars.lazyframe.frame import LazyFrame
    from polars.lazyframe.opt_flags import QueryOptFlags
    from polars.lazyframe.query_result import QueryResult


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


def _apply_retries_deprecation(
    retries: int | None, storage_options: StorageOptionsDict | None
) -> StorageOptionsDict | None:
    if retries is not None:
        msg = "the `retries` parameter was deprecated in 1.37.1; specify 'max_retries' in `storage_options` instead."
        issue_deprecation_warning(msg)
        storage_options = storage_options or {}
        storage_options["max_retries"] = retries
    return storage_options


class Engine(ABC):
    """
    Base class for query execution engines.

    Subclass this to plug a new backend into Polars.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier."""

    @property
    def plan_engine(self) -> str:
        """Engine name used to render query plans."""
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @overload
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: Literal[False] = ...,
        post_opt_callback: PostOptCallback | None = ...,
    ) -> DataFrame: ...

    @overload
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: Literal[True],
        post_opt_callback: PostOptCallback | None = ...,
    ) -> InProcessQuery: ...

    @overload
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: bool,
        post_opt_callback: PostOptCallback | None = ...,
    ) -> DataFrame | InProcessQuery: ...

    @abstractmethod
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: bool = False,
        post_opt_callback: PostOptCallback | None = None,
    ) -> DataFrame | InProcessQuery:
        """
        Execute `lf`, returning a `DataFrame` or a background query handle.

        Parameters
        ----------
        lf
            The query to execute.
        optimizations
            Optimization passes to apply.
        background
            Run in the background and return an `InProcessQuery`.
        post_opt_callback
            Internal post-optimization callback.
        """

    @abstractmethod
    def execute(self, lf: LazyFrame, *, optimizations: QueryOptFlags) -> QueryResult:
        """
        Execute the query into a `QueryResult`.

        This method of materializing a `LazyFrame` makes no guarantees as to where
        the result is materialized. This can be on the GPU for the GPU-engine,
        on the cluster or remote storage for the distributed engine and the streaming
        engine could spill the result if it needed to.

        The `QueryResult` can always be consumed as a new `LazyFrame` by calling `.lazy`
        """

    def collect_async(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        gevent: bool = False,
    ) -> AsyncResult[DataFrame]:
        """Execute `lf` asynchronously."""
        msg = f"`collect_async` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def collect_batches(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        maintain_order: bool = True,
        chunk_size: int | None = None,
        lazy: bool = False,
    ) -> Iterator[DataFrame]:
        """Execute `lf`, yielding its result in batches."""
        msg = f"`collect_batches` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def collect_all(
        self, lfs: Iterable[LazyFrame], *, optimizations: QueryOptFlags
    ) -> list[DataFrame]:
        """Execute several queries, potentially in parallel."""
        msg = f"`collect_all` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def collect_all_async(
        self,
        lfs: Iterable[LazyFrame],
        *,
        optimizations: QueryOptFlags,
        gevent: bool = False,
    ) -> AsyncResult[list[DataFrame]]:
        """Execute several queries asynchronously."""
        msg = f"`collect_all_async` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def sink_parquet(
        self,
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
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        metadata: ParquetMetadata | None,
        arrow_schema: ArrowSchemaExportable | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> LazyFrame | None:
        """See :meth:`polars.LazyFrame.sink_parquet`."""
        msg = f"`sink_parquet` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def sink_ipc(
        self,
        lf: LazyFrame,
        path: str | Path | IO[bytes] | PartitionBy,
        *,
        compression: IpcCompression | None,
        compat_level: CompatLevel | None,
        record_batch_size: int | None,
        maintain_order: bool,
        storage_options: StorageOptionsDict | None,
        credential_provider: CredentialProviderFunction | Literal["auto"] | None,
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
        _record_batch_statistics: bool,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> LazyFrame | None:
        """See :meth:`polars.LazyFrame.sink_ipc`."""
        msg = f"`sink_ipc` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def sink_csv(
        self,
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
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> LazyFrame | None:
        """See :meth:`polars.LazyFrame.sink_csv`."""
        msg = f"`sink_csv` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def sink_ndjson(
        self,
        lf: LazyFrame,
        path: str | Path | IO[bytes] | IO[str] | PartitionBy,
        *,
        compression: Literal["uncompressed", "gzip", "zstd"],
        compression_level: int | None,
        check_extension: bool,
        maintain_order: bool,
        storage_options: StorageOptionsDict | None,
        credential_provider: CredentialProviderFunction | Literal["auto"] | None,
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> LazyFrame | None:
        """See :meth:`polars.LazyFrame.sink_ndjson`."""
        msg = f"`sink_ndjson` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)

    def sink_batches(
        self,
        lf: LazyFrame,
        function: Callable[[DataFrame], bool | None],
        *,
        chunk_size: int | None,
        maintain_order: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> LazyFrame | None:
        """See :meth:`polars.LazyFrame.sink_batches`."""
        msg = f"`sink_batches` is not supported by {type(self).__name__}"
        raise NotImplementedError(msg)


class _LocalEngine(Engine):
    """Base for in-process engines backed by `PyLazyFrame`."""

    _name: ClassVar[str]

    # Subclasses that do not call `super().__init__()` still see the default value.
    monitoring: bool | None = None
    """Whether queries are monitored (``None`` uses the configured default)."""

    def __init__(self, *, monitoring: bool | None = None) -> None:
        if monitoring:
            from polars._utils.monitoring import activate_monitoring

            activate_monitoring()
        self.monitoring = monitoring

    def __repr__(self) -> str:
        args = "" if self.monitoring is None else f"monitoring={self.monitoring!r}"
        return f"{type(self).__name__}({args})"

    @property
    def name(self) -> str:
        """Name of the engine."""
        return self._name

    def _with_monitoring(self, optimizations: QueryOptFlags) -> QueryOptFlags:
        """Register the query observer when enabled and update `optimizations`."""
        from polars._utils.monitoring import monitoring_enabled_globally

        monitor = (
            monitoring_enabled_globally()
            if self.monitoring is None
            else self.monitoring
        )
        if monitor:
            import polars._plr as plr

            plr.set_query_monitoring(True)

        optimizations = optimizations.__copy__()
        optimizations._pyoptflags.query_monitoring = monitor
        return optimizations

    def execute(self, lf: LazyFrame, *, optimizations: QueryOptFlags) -> QueryResult:
        df = self.collect(lf, optimizations=optimizations)
        return SingleNodeQueryResult(df)  # type: ignore[arg-type]

    def _post_opt_callback(
        self,
        *,
        background: bool,  # noqa: ARG002
        eager: bool,  # noqa: ARG002
    ) -> PostOptCallback | None:
        return None

    @overload
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: Literal[False] = ...,
        post_opt_callback: PostOptCallback | None = ...,
    ) -> DataFrame: ...

    @overload
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: Literal[True],
        post_opt_callback: PostOptCallback | None = ...,
    ) -> InProcessQuery: ...

    @overload
    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: bool,
        post_opt_callback: PostOptCallback | None = ...,
    ) -> DataFrame | InProcessQuery: ...

    def collect(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: bool = False,
        post_opt_callback: PostOptCallback | None = None,
    ) -> DataFrame | InProcessQuery:
        callback = self._post_opt_callback(
            background=background, eager=optimizations._pyoptflags.eager
        )
        optimizations = self._with_monitoring(optimizations)

        ldf = lf._ldf.with_optimizations(optimizations._pyoptflags)
        if background:
            issue_unstable_warning("background mode is considered unstable.")
            return InProcessQuery(ldf.collect_concurrently())

        if post_opt_callback is not None:
            callback = post_opt_callback
        return wrap_df(ldf.collect(self.name, callback))

    def collect_async(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        gevent: bool = False,
    ) -> AsyncResult[DataFrame]:
        if self.name == "streaming":
            issue_unstable_warning("streaming mode is considered unstable.")

        optimizations = self._with_monitoring(optimizations)
        ldf = lf._ldf.with_optimizations(optimizations._pyoptflags)
        result: AsyncResult[DataFrame] = (
            _GeventDataFrameResult() if gevent else _AioDataFrameResult()
        )
        ldf.collect_with_callback(self.name, result._callback)
        return result

    def collect_batches(
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        maintain_order: bool = True,
        chunk_size: int | None = None,
        lazy: bool = False,
    ) -> Iterator[DataFrame]:
        optimizations = self._with_monitoring(optimizations)
        ldf = lf._ldf.with_optimizations(optimizations._pyoptflags)
        return _CollectBatches(
            ldf.collect_batches(
                engine=self.name,
                maintain_order=maintain_order,
                chunk_size=chunk_size,
                lazy=lazy,
            )
        )

    def collect_all(
        self, lfs: Iterable[LazyFrame], *, optimizations: QueryOptFlags
    ) -> list[DataFrame]:
        import polars._plr as plr

        optimizations = self._with_monitoring(optimizations)
        out = plr.collect_all(
            [lf._ldf for lf in lfs], self.name, optimizations._pyoptflags
        )
        return [wrap_df(pydf) for pydf in out]

    def collect_all_async(
        self,
        lfs: Iterable[LazyFrame],
        *,
        optimizations: QueryOptFlags,
        gevent: bool = False,
    ) -> AsyncResult[list[DataFrame]]:
        import polars._plr as plr

        optimizations = self._with_monitoring(optimizations)
        result: AsyncResult[list[DataFrame]] = (
            _GeventDataFrameResult() if gevent else _AioDataFrameResult()
        )
        plr.collect_all_with_callback(
            [lf._ldf for lf in lfs],
            self.name,
            optimizations._pyoptflags,
            result._callback_all,
        )
        return result

    def _finish_sink(
        self, ldf_py: PyLazyFrame, *, lazy: bool, optimizations: QueryOptFlags
    ) -> LazyFrame | None:
        """Return a lazy sink plan, or execute it."""
        lf = wrap_ldf(ldf_py)
        if lazy:
            return lf
        self.collect(lf, optimizations=optimizations)
        return None

    def sink_parquet(
        self,
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
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        metadata: ParquetMetadata | None,
        arrow_schema: ArrowSchemaExportable | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> LazyFrame | None:
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

        storage_options = _apply_retries_deprecation(retries, storage_options)

        credential_provider_builder = _init_credential_provider_builder(
            credential_provider, path, storage_options, "sink_parquet"
        )
        del credential_provider

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
        return self._finish_sink(ldf_py, lazy=lazy, optimizations=optimizations)

    def sink_ipc(
        self,
        lf: LazyFrame,
        path: str | Path | IO[bytes] | PartitionBy,
        *,
        compression: IpcCompression | None,
        compat_level: CompatLevel | None,
        record_batch_size: int | None,
        maintain_order: bool,
        storage_options: StorageOptionsDict | None,
        credential_provider: CredentialProviderFunction | Literal["auto"] | None,
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
        _record_batch_statistics: bool,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> LazyFrame | None:
        from polars.interchange.protocol import CompatLevel
        from polars.io.cloud.credential_provider._builder import (
            _init_credential_provider_builder,
        )
        from polars.io.partition import _SinkOptions

        storage_options = _apply_retries_deprecation(retries, storage_options)

        credential_provider_builder = _init_credential_provider_builder(
            credential_provider, path, storage_options, "sink_ipc"
        )
        del credential_provider

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
        return self._finish_sink(ldf_py, lazy=lazy, optimizations=optimizations)

    def sink_csv(
        self,
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
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> LazyFrame | None:
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
        del credential_provider

        target = _to_sink_target(path)

        storage_options = _apply_retries_deprecation(retries, storage_options)

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
        return self._finish_sink(ldf_py, lazy=lazy, optimizations=optimizations)

    def sink_ndjson(
        self,
        lf: LazyFrame,
        path: str | Path | IO[bytes] | IO[str] | PartitionBy,
        *,
        compression: Literal["uncompressed", "gzip", "zstd"],
        compression_level: int | None,
        check_extension: bool,
        maintain_order: bool,
        storage_options: StorageOptionsDict | None,
        credential_provider: CredentialProviderFunction | Literal["auto"] | None,
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> LazyFrame | None:
        from polars.io.cloud.credential_provider._builder import (
            _init_credential_provider_builder,
        )
        from polars.io.partition import _SinkOptions

        storage_options = _apply_retries_deprecation(retries, storage_options)

        credential_provider_builder = _init_credential_provider_builder(
            credential_provider, path, storage_options, "sink_ndjson"
        )
        del credential_provider

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
        return self._finish_sink(ldf_py, lazy=lazy, optimizations=optimizations)

    def sink_batches(
        self,
        lf: LazyFrame,
        function: Callable[[DataFrame], bool | None],
        *,
        chunk_size: int | None,
        maintain_order: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> LazyFrame | None:
        from polars._utils.wrap import wrap_df

        def _wrap(pydf: Any) -> bool:
            return bool(function(wrap_df(pydf)))

        ldf_py = lf._ldf.sink_batches(
            function=_wrap,
            maintain_order=maintain_order,
            chunk_size=chunk_size,
        )
        return self._finish_sink(ldf_py, lazy=lazy, optimizations=optimizations)


class _CollectBatches:
    """Iterator over the batches produced by `Engine.collect_batches`."""

    def __init__(self, inner: PyCollectBatches) -> None:
        self._inner = inner

    def __iter__(self) -> _CollectBatches:
        return self

    def __next__(self) -> DataFrame:
        return wrap_df(next(self._inner))

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self._inner.__arrow_c_stream__(requested_schema)


class _AutoEngine(_LocalEngine):
    _name = "auto"


class InMemoryEngine(_LocalEngine):
    """
    The in-memory engine.

    Parameters
    ----------
    monitoring : bool, default None
        Whether to monitor queries run by this engine. ``None`` uses the setting from
        :meth:`Config.enable_monitoring`; ``True`` or ``False`` overrides it. Setting
        this to ``True`` requires the ``polars-cloud`` package.
    """

    _name = "in-memory"


class StreamingEngine(_LocalEngine):
    """
    The streaming engine.

    Parameters
    ----------
    monitoring : bool, default None
        Whether to monitor queries run by this engine. ``None`` uses the setting from
        :meth:`Config.enable_monitoring`; ``True`` or ``False`` overrides it. Setting
        this to ``True`` requires the ``polars-cloud`` package.

    Examples
    --------
    Monitor a single query, regardless of the configured default:

    >>> lf.collect(engine=pl.StreamingEngine(monitoring=True))  # doctest: +SKIP
    """

    _name = "streaming"


class GPUEngine(_LocalEngine):
    """
    Configuration options for the GPU execution engine.

    Use this if you want control over details of the execution.

    Parameters
    ----------
    device : int, default None
        Select the GPU used to run the query. If not provided, the
        query uses the current CUDA device.
    memory_resource : rmm.mr.DeviceMemoryResource, default None
        Provide a memory resource for GPU memory allocations.

        .. warning::
           If passing a `memory_resource`, you must ensure that it is valid
           for the selected `device`. See the `RMM documentation
           <https://github.com/rapidsai/rmm?tab=readme-ov-file#multiple-devices>`_
           for more details.

    raise_on_fail : bool, default False
        If True, do not fall back to the Polars CPU engine if the GPU
        engine cannot execute the query, but instead raise an error.

    """

    _name = "gpu"

    device: int | None
    """Device on which to run query."""
    memory_resource: DeviceMemoryResource | None
    """Memory resource to use for device allocations."""
    raise_on_fail: bool
    """
    Whether unsupported queries should raise an error, rather than falling
    back to the CPU engine.
    """
    config: Mapping[str, Any]
    """Additional configuration options for the engine."""

    def __init__(
        self,
        *,
        device: int | None = None,
        memory_resource: Any | None = None,
        raise_on_fail: bool = False,
        monitoring: bool = False,
        **kwargs: Any,
    ) -> None:
        # We do want a named param for `monitoring`, because otherwise it will silently
        # end up in `kwargs`
        if monitoring:
            msg = "query monitoring is not supported by the GPU engine"
            raise NotImplementedError(msg)
        super().__init__(monitoring=False)

        self.device = device
        self.memory_resource = memory_resource
        # Avoids need for changes in cudf-polars
        kwargs["raise_on_fail"] = raise_on_fail
        self.config = kwargs

    def _post_opt_callback(
        self, *, background: bool, eager: bool
    ) -> Callable[[Any, int | None], None] | None:
        if background:
            issue_warning(
                "GPU engine does not support background collection, disabling GPU engine.",
                category=UserWarning,
            )
            return None
        if eager:
            # Don't run on GPU in _eager mode (but don't warn)
            return None

        cudf_polars = import_optional(
            "cudf_polars",
            err_prefix="GPU engine requested, but required package",
            install_message=(
                "Please install using the command "
                "`pip install cudf-polars-cu12` "
                "(CUDA 12 is required for RAPIDS cuDF v25.08 and later). "
                "If your system has a CUDA 11 driver, install with "
                "`pip install cudf-polars-cu11==25.06` "
            ),
        )
        return partial(cudf_polars.execute_with_cudf, config=self)
