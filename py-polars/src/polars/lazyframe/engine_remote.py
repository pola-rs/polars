"""
Remote query execution through Polars Cloud.

Cloud integration is isolated here; the optional `polars_cloud` dependency is
loaded when a `RemoteEngine` is constructed.
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, Literal, get_args

from polars._dependencies import import_optional
from polars._typing import EngineTypeName, ScalingMode
from polars._utils.various import qualified_type_name
from polars._warnings import issue_warning
from polars.lazyframe.engine import Engine, StreamingEngine

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import polars_cloud as pc

    from polars._typing import (
        ArrowSchemaExportable,
        CsvQuoteStyle,
        IpcCompression,
        ParquetCompression,
        ParquetMetadata,
        PlanTypePreference,
        PostOptCallback,
        StorageOptionsDict,
        SyncOnCloseMethod,
    )
    from polars.dataframe import DataFrame
    from polars.interchange.protocol import CompatLevel
    from polars.io.cloud import CredentialProviderFunction
    from polars.io.partition import PartitionBy, SinkedPathsCallback
    from polars.lazyframe.frame import LazyFrame
    from polars.lazyframe.in_process import InProcessQuery
    from polars.lazyframe.opt_flags import QueryOptFlags
    from polars.lazyframe.query_result import QueryResult

_SCALING_MODES = get_args(ScalingMode)
_WORKER_ENGINE_NAMES = get_args(EngineTypeName)


class RemoteEngine(Engine):
    """
    Execute queries remotely on Polars Cloud.

    Use this engine with :meth:`LazyFrame.execute`, a `LazyFrame.sink_*` method,
    or as the default through :meth:`Config.set_engine_affinity`.

    Requires the `polars_cloud` package.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    context
        Compute context. Defaults to the current `polars_cloud` session.
    scaling_mode : {'auto', 'single-node', 'distributed'}
        Whether to run the query on a single node or distributed over the cluster.
        If `'auto'`, a query runs distributed if the cluster has more than one node.
    engine : {'auto', 'in-memory', 'streaming', 'gpu'}
        Preferred worker engine. This is a hint and also controls plan rendering.
    plan_type : {'dot', 'plain'}
        Query-plan rendering format.
    n_retries
        Number of retries per failed stage.
    labels
        Labels attached to the query. Missing labels are created automatically.
    **kwargs
        Additional options forwarded to the distributed planner, such as
        `max_workers`, `min_workers`, `shuffle_format` or `partitions_per_worker`.
        Passing any of these is incompatible with `scaling_mode='single-node'`.

    Examples
    --------
    >>> import polars_cloud as pc  # doctest: +SKIP
    >>> engine = pl.RemoteEngine(  # doctest: +SKIP
    ...     pc.ComputeContext(cpus=16, memory=64),
    ...     scaling_mode="distributed",
    ...     max_workers=8,
    ... )
    >>> lf.sink_parquet("s3://my-destination/", engine=engine)  # doctest: +SKIP

    Make it the default for every query in a scope:

    >>> with pl.Config(engine_affinity=engine):  # doctest: +SKIP
    ...     lf.sink_parquet("s3://my-destination/")
    """

    context: pc.ClientContext | None
    """Compute context in which queries are executed."""
    scaling_mode: ScalingMode
    """Whether the query runs on a single node or distributed over the cluster."""
    engine: EngineTypeName
    """Engine the workers are asked to prefer."""
    plan_type: PlanTypePreference
    """How query plans are rendered."""
    n_retries: int
    """How often a stage should be retried on failure."""
    labels: list[str] | None
    """Labels attached to the query."""
    config: Mapping[str, Any]
    """Additional options forwarded to the distributed planner."""

    def __init__(
        self,
        context: pc.ClientContext | None = None,
        *,
        scaling_mode: ScalingMode = "auto",
        engine: EngineTypeName = "auto",
        plan_type: PlanTypePreference = "dot",
        n_retries: int = 0,
        labels: list[str] | str | None = None,
        **kwargs: Any,
    ) -> None:
        if scaling_mode not in _SCALING_MODES:
            msg = f"invalid `scaling_mode` {scaling_mode!r}"
            raise ValueError(msg)
        if engine not in _WORKER_ENGINE_NAMES:
            msg = f"Invalid engine argument {engine=}"
            raise ValueError(msg)
        if scaling_mode == "single-node" and kwargs:
            msg = (
                f"distributed options {sorted(kwargs)!r} are not supported with "
                "`scaling_mode='single-node'`"
            )
            raise ValueError(msg)

        # fail here rather than deep inside a sink
        import_optional(
            "polars_cloud",
            err_prefix="remote engine requested, but required package",
            install_message="Please install using the command `pip install polars-cloud`",
        )

        self.context = context
        self.scaling_mode = scaling_mode
        self.engine = engine
        self.plan_type = plan_type
        self.n_retries = n_retries
        self.labels = [labels] if isinstance(labels, str) else labels
        self.config = kwargs

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "remote"

    @property
    def plan_engine(self) -> str:
        """Preferred worker engine used to render query plans."""
        return self.engine

    def _target(self, lf: LazyFrame) -> pc.LazyFrameRemote | pc.ExecuteRemote:
        """Return the `polars_cloud` object that executes `lf`."""
        remote = lf.remote(
            self.context,
            plan_type=self.plan_type,
            n_retries=self.n_retries,
            engine=self.engine,
            scaling_mode=self.scaling_mode,
        )
        if self.labels:
            remote = remote.labels(self.labels)
        if self.config:
            return remote.distributed(**self.config)
        return remote  # let `polars_cloud` resolve the scaling mode

    # -- Execution ----------------------------------------------------------------

    def execute(self, lf: LazyFrame, *, optimizations: QueryOptFlags) -> QueryResult:
        """See :meth:`polars.LazyFrame.execute`."""
        return self._target(lf).execute(optimizations=optimizations)  # type: ignore[no-any-return]

    def collect(  # type: ignore[override]
        self,
        lf: LazyFrame,
        *,
        optimizations: QueryOptFlags,
        background: bool = False,
        post_opt_callback: PostOptCallback | None = None,
    ) -> DataFrame | InProcessQuery:
        """
        Collect `lf` remotely, then transfer the result to this machine.

        Prefer `LazyFrame.execute` or a sink to avoid transferring the full
        result.
        """
        issue_warning(
            "collecting a remote query transfers the entire result to this machine; "
            "use `LazyFrame.execute` or a `sink_*` method to avoid the round-trip.",
            category=UserWarning,
        )
        result = self.execute(lf, optimizations=optimizations)
        return StreamingEngine().collect(
            result.lazy(),
            optimizations=optimizations,
            background=background,
            post_opt_callback=post_opt_callback,
        )

    # -- Sinks --------------------------------------------------------------------

    @staticmethod
    def _sink_uri(path: Any) -> str | PartitionBy:
        """Validate a Polars Cloud sink target."""
        from polars.io.partition import PartitionBy

        if not isinstance(path, (str, PartitionBy)):
            msg = (
                "the remote engine can only sink to a URI or a `PartitionBy`, got "
                f"{qualified_type_name(path)!r}"
            )
            raise TypeError(msg)
        return path

    @staticmethod
    def _reject_if_set(**kwargs: Any) -> None:
        """Reject options unsupported by Polars Cloud."""
        for name, value in kwargs.items():
            if value:
                msg = f"`{name}` is not supported by the remote engine"
                raise ValueError(msg)

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
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_parquet`."""
        self._reject_if_set(
            lazy=lazy,
            mkdir=mkdir,
            sync_on_close=sync_on_close,
            retries=retries,
            sinked_paths_callback=sinked_paths_callback,
        )
        self._target(lf).sink_parquet(
            self._sink_uri(path),
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
            row_group_size=row_group_size,
            data_page_size=data_page_size,
            maintain_order=maintain_order,
            storage_options=storage_options,
            credential_provider=credential_provider,
            metadata=metadata,
            arrow_schema=arrow_schema,
            optimizations=optimizations,
        ).await_result()
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
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
        _record_batch_statistics: bool,
        sinked_paths_callback: SinkedPathsCallback | None,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_ipc`."""
        self._reject_if_set(
            lazy=lazy,
            mkdir=mkdir,
            sync_on_close=sync_on_close,
            retries=retries,
            record_batch_size=record_batch_size,
            _record_batch_statistics=_record_batch_statistics,
            sinked_paths_callback=sinked_paths_callback,
            # Polars Cloud only supports `maintain_order=True`.
            maintain_order=not maintain_order,
        )
        self._target(lf).sink_ipc(
            self._sink_uri(path),
            compression=compression,
            compat_level=compat_level,
            storage_options=storage_options,
            credential_provider=credential_provider,
            optimizations=optimizations,
        ).await_result()
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
        retries: int | None,
        sync_on_close: SyncOnCloseMethod | None,
        mkdir: bool,
        lazy: bool,
        optimizations: QueryOptFlags,
    ) -> None:
        """See :meth:`polars.LazyFrame.sink_csv`."""
        self._reject_if_set(
            lazy=lazy,
            mkdir=mkdir,
            sync_on_close=sync_on_close,
            retries=retries,
            compression=compression != "uncompressed",
            compression_level=compression_level,
            check_extension=not check_extension,
            # Polars Cloud only supports `maintain_order=True`.
            maintain_order=not maintain_order,
        )
        self._target(lf).sink_csv(
            self._sink_uri(path),
            include_bom=include_bom,
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
            storage_options=storage_options,
            credential_provider=credential_provider,
            optimizations=optimizations,
        ).await_result()
        return None
