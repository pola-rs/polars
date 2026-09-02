"""Query execution engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import IO, TYPE_CHECKING, Any, ClassVar, Literal, overload

from polars._dependencies import import_optional
from polars._utils.async_ import _AioDataFrameResult, _GeventDataFrameResult
from polars._utils.unstable import issue_unstable_warning
from polars._utils.wrap import wrap_df
from polars._warnings import issue_warning
from polars.lazyframe.in_process import InProcessQuery
from polars.lazyframe.query_result import SingleNodeQueryResult
from polars.lazyframe.sink_plan import (
    _sink_batches_plan,
    _sink_csv_plan,
    _sink_ipc_plan,
    _sink_ndjson_plan,
    _sink_parquet_plan,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from rmm.mr import DeviceMemoryResource  # type: ignore[import-not-found]

    from polars._plr import PyCollectBatches
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
    ) -> _CollectBatches:
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
        sync_on_close: SyncOnCloseMethod | None,
        metadata: ParquetMetadata | None,
        arrow_schema: ArrowSchemaExportable | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> None:
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
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
        _record_batch_statistics: bool,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> None:
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
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
    ) -> None:
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
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
    ) -> None:
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
        optimizations: QueryOptFlags,
    ) -> None:
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
    ) -> _CollectBatches:
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
        sync_on_close: SyncOnCloseMethod | None,
        metadata: ParquetMetadata | None,
        arrow_schema: ArrowSchemaExportable | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_parquet`."""
        self.collect(
            _sink_parquet_plan(
                lf,
                path,
                compression=compression,
                compression_level=compression_level,
                statistics=statistics,
                row_group_size=row_group_size,
                data_page_size=data_page_size,
                maintain_order=maintain_order,
                storage_options=storage_options,
                credential_provider=credential_provider,
                sync_on_close=sync_on_close,
                metadata=metadata,
                arrow_schema=arrow_schema,
                mkdir=mkdir,
                sinked_paths_callback=sinked_paths_callback,
            ),
            optimizations=optimizations,
        )
        return None

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
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
        _record_batch_statistics: bool,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_ipc`."""
        self.collect(
            _sink_ipc_plan(
                lf,
                path,
                compression=compression,
                compat_level=compat_level,
                record_batch_size=record_batch_size,
                maintain_order=maintain_order,
                storage_options=storage_options,
                credential_provider=credential_provider,
                sync_on_close=sync_on_close,
                mkdir=mkdir,
                _record_batch_statistics=_record_batch_statistics,
                sinked_paths_callback=sinked_paths_callback,
            ),
            optimizations=optimizations,
        )
        return None

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
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_csv`."""
        self.collect(
            _sink_csv_plan(
                lf,
                path,
                include_bom=include_bom,
                compression=compression,
                compression_level=compression_level,
                check_extension=check_extension,
                include_header=include_header,
                separator=separator,
                line_terminator=line_terminator,
                quote_char=quote_char,
                batch_size=batch_size,
                datetime_format=datetime_format,
                date_format=date_format,
                time_format=time_format,
                float_scientific=float_scientific,
                float_precision=float_precision,
                decimal_comma=decimal_comma,
                null_value=null_value,
                quote_style=quote_style,
                maintain_order=maintain_order,
                storage_options=storage_options,
                credential_provider=credential_provider,
                sync_on_close=sync_on_close,
                mkdir=mkdir,
            ),
            optimizations=optimizations,
        )
        return None

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
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        optimizations: QueryOptFlags,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_ndjson`."""
        self.collect(
            _sink_ndjson_plan(
                lf,
                path,
                compression=compression,
                compression_level=compression_level,
                check_extension=check_extension,
                maintain_order=maintain_order,
                storage_options=storage_options,
                credential_provider=credential_provider,
                sync_on_close=sync_on_close,
                mkdir=mkdir,
            ),
            optimizations=optimizations,
        )
        return None

    def sink_batches(
        self,
        lf: LazyFrame,
        function: Callable[[DataFrame], bool | None],
        *,
        chunk_size: int | None,
        maintain_order: bool,
        optimizations: QueryOptFlags,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_batches`."""
        self.collect(
            _sink_batches_plan(
                lf,
                function,
                chunk_size=chunk_size,
                maintain_order=maintain_order,
            ),
            optimizations=optimizations,
        )
        return None


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
            err_suffix="could not be imported",
            install_message=(
                "Please install the cuDF Polars distribution matching your CUDA "
                "version. See the GPU support documentation for installation "
                "instructions: https://docs.pola.rs/user-guide/gpu-support/."
            ),
        )
        return partial(cudf_polars.execute_with_cudf, config=self)
