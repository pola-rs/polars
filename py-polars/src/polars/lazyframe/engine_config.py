from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, NamedTuple

from polars._dependencies import import_optional

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import get_engine_affinity

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal, NoReturn

    import polars_cloud as pc
    from rmm.mr import DeviceMemoryResource  # type: ignore[import-not-found]

    from polars._typing import EngineType, EngineTypeName
    from polars.lazyframe.frame import LazyFrame
    from polars.lazyframe.opt_flags import QueryOptFlags
    from polars.lazyframe.query_result import QueryResult


class GPUEngine:
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
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.memory_resource = memory_resource
        # Avoids need for changes in cudf-polars
        kwargs["raise_on_fail"] = raise_on_fail
        self.config = kwargs


class RemoteEngine:
    """
    Configuration options for remote execution on Polars Cloud.

    Pass an instance as the `engine` argument of :meth:`LazyFrame.execute` or of a
    `LazyFrame.sink_*` method, or make it the default for all queries with
    :meth:`Config.set_engine_affinity`.

    Requires the `polars_cloud` package to be installed.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    context
        Compute context in which queries are executed. If not given, the default
        context of the `polars_cloud` session is used.
    scaling_mode : {'auto', 'single-node', 'distributed'}
        Whether to run the query on a single node or distributed over the cluster.
        If set to `'auto'`, a query runs distributed if the cluster has more than
        one node.
    engine : {'auto', 'in-memory', 'streaming', 'gpu'}
        Hint that tells the workers which engine to prefer. It does not have to be
        respected.
    plan_type : {'dot', 'plain'}
        Whether to render query plans as a dot diagram or as plain text.
    n_retries
        How often a stage should be retried on failure.
    labels
        Labels to attach to the query. Labels are implicitly created.
    **kwargs
        Additional options forwarded to the distributed planner, such as
        `max_workers`, `min_workers`, `shuffle_format` or `partitions_per_worker`.
        See the `polars_cloud` documentation for the full list. Passing any of
        these is incompatible with `scaling_mode='single-node'`.

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
    scaling_mode: Literal["auto", "single-node", "distributed"]
    """Whether the query runs on a single node or distributed over the cluster."""
    engine: EngineTypeName
    """Engine the workers should prefer."""
    plan_type: Literal["dot", "plain"]
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
        scaling_mode: Literal["auto", "single-node", "distributed"] = "auto",
        engine: EngineTypeName = "auto",
        plan_type: Literal["dot", "plain"] = "dot",
        n_retries: int = 0,
        labels: list[str] | str | None = None,
        **kwargs: Any,
    ) -> None:
        if scaling_mode not in ("auto", "single-node", "distributed"):
            msg = f"invalid `scaling_mode` {scaling_mode!r}"
            raise ValueError(msg)
        if scaling_mode == "single-node" and kwargs:
            msg = (
                f"distributed options {sorted(kwargs)!r} are not supported with "
                "`scaling_mode='single-node'`"
            )
            raise ValueError(msg)

        self.context = context
        self.scaling_mode = scaling_mode
        self.engine = engine
        self.plan_type = plan_type
        self.n_retries = n_retries
        self.labels = [labels] if isinstance(labels, str) else labels
        self.config = kwargs

    def _target(self, lf: LazyFrame) -> Any:
        """Return the `polars_cloud` object that executes `lf` on this engine."""
        import_optional(
            "polars_cloud",
            err_prefix="remote engine requested, but required package",
            install_message=(
                "Please install using the command `pip install polars-cloud`"
            ),
        )
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

    def _execute(self, lf: LazyFrame, optimizations: QueryOptFlags) -> QueryResult:
        """Start `lf` on Polars Cloud and return a handle to the running query."""
        return self._target(lf).execute(optimizations=optimizations)  # type: ignore[no-any-return]

    def _sink(self, lf: LazyFrame, method: str, params: Mapping[str, Any]) -> None:
        """
        Run a `LazyFrame.sink_*` method on Polars Cloud, blocking until it completes.

        `params` is the calling method's `locals()`; which of its arguments are
        forwarded and which have no remote equivalent is described by
        `_REMOTE_SINK_PARAMS`.
        """
        from polars._utils.various import qualified_type_name
        from polars.io.partition import PartitionBy

        spec = _REMOTE_SINK_PARAMS[method]

        path = params["path"]
        if not isinstance(path, (str, PartitionBy)):
            msg = (
                "the remote engine can only sink to a URI or a `PartitionBy`, got "
                f"{qualified_type_name(path)!r}"
            )
            raise TypeError(msg)

        for name, default in spec.unsupported.items():
            if params[name] != default:
                msg = f"`{name}` is not supported by the remote engine"
                raise ValueError(msg)

        kwargs = {name: params[name] for name in spec.forward}
        getattr(self._target(lf), method)(path, **kwargs).await_result()

    def _raise_unsupported(self, method: str) -> NoReturn:
        """Raise for an operation that the remote engine cannot run."""
        msg = (
            f"`{method}` is not supported by the remote engine\n\n"
            "Use `LazyFrame.execute`, `LazyFrame.sink_parquet`, "
            "`LazyFrame.sink_csv` or `LazyFrame.sink_ipc` to run this query on "
            "Polars Cloud, or pass an explicit `engine=` to run it locally."
        )
        raise NotImplementedError(msg)


class _RemoteSinkParams(NamedTuple):
    """How the arguments of one `LazyFrame.sink_*` method map onto Polars Cloud."""

    forward: tuple[str, ...]
    """Arguments passed on to the `polars_cloud` sink method unchanged."""
    unsupported: Mapping[str, Any]
    """
    Arguments with no remote equivalent, mapped to their local default. Passing
    anything else is an error rather than a silent no-op.
    """


# Keep the defaults in sync with the signatures of the `LazyFrame.sink_*` methods.
_REMOTE_SINK_PARAMS: Mapping[str, _RemoteSinkParams] = {
    "sink_parquet": _RemoteSinkParams(
        forward=(
            "compression",
            "compression_level",
            "statistics",
            "row_group_size",
            "data_page_size",
            "maintain_order",
            "storage_options",
            "credential_provider",
            "metadata",
            "arrow_schema",
            "optimizations",
        ),
        unsupported={
            "lazy": False,
            "mkdir": False,
            "sync_on_close": None,
            "retries": None,
            "_sinked_paths_callback": None,
        },
    ),
    "sink_ipc": _RemoteSinkParams(
        forward=(
            "compression",
            "compat_level",
            "storage_options",
            "credential_provider",
            "optimizations",
        ),
        unsupported={
            "lazy": False,
            "mkdir": False,
            "sync_on_close": None,
            "retries": None,
            "record_batch_size": None,
            "_record_batch_statistics": False,
            "maintain_order": True,
        },
    ),
    "sink_csv": _RemoteSinkParams(
        forward=(
            "include_bom",
            "include_header",
            "separator",
            "line_terminator",
            "quote_char",
            "batch_size",
            "datetime_format",
            "date_format",
            "time_format",
            "float_scientific",
            "float_precision",
            "decimal_comma",
            "null_value",
            "quote_style",
            "storage_options",
            "credential_provider",
            "optimizations",
        ),
        unsupported={
            "lazy": False,
            "mkdir": False,
            "sync_on_close": None,
            "retries": None,
            "compression": "uncompressed",
            "compression_level": None,
            "check_extension": True,
            "maintain_order": True,
        },
    ),
}


_ENGINE_AFFINITY_OVERRIDE: GPUEngine | RemoteEngine | None = None


def get_engine_affinity_override() -> GPUEngine | RemoteEngine | None:
    """Return the object-valued default engine, if one is configured."""
    return _ENGINE_AFFINITY_OVERRIDE


def set_engine_affinity_override(engine: GPUEngine | RemoteEngine | None) -> None:
    """Set (or clear, with `None`) the object-valued default engine."""
    global _ENGINE_AFFINITY_OVERRIDE
    _ENGINE_AFFINITY_OVERRIDE = engine


def _select_engine(engine: EngineType) -> EngineType:
    """Resolve `"auto"` to the engine affinity, which may be an engine object."""
    if engine != "auto":
        return engine
    # the env var that rust reads can only name an engine, never hold an object
    override = get_engine_affinity_override()
    return get_engine_affinity() if override is None else override


def _select_local_engine(engine: EngineType, method: str) -> EngineType:
    """Resolve `engine` for a `method` that has no remote implementation."""
    engine = _select_engine(engine)
    if isinstance(engine, RemoteEngine):
        engine._raise_unsupported(method)
    return engine


def _select_plan_engine(engine: EngineType) -> EngineType:
    """Resolve `engine` for query plan rendering, which always happens locally."""
    engine = _select_engine(engine)
    if isinstance(engine, RemoteEngine):
        # show the plan for the engine the remote workers would prefer
        return engine.engine
    return engine


def _select_collect_engine(engine: EngineType, *, eager: bool) -> EngineType:
    """
    Resolve `engine` for `LazyFrame.collect`, which is local-only.

    Eager `DataFrame` operations are built on `collect`, so an `eager` query keeps
    running locally instead of raising when a remote engine is the default.
    """
    engine = _select_engine(engine)
    if isinstance(engine, RemoteEngine):
        if not eager:
            engine._raise_unsupported("LazyFrame.collect")
        return _default_local_engine()
    return engine


def _default_local_engine() -> EngineType:
    """
    Return the default local engine, ignoring any object-valued affinity.

    Never returns `"auto"`, which would re-enter the affinity lookup; rust maps
    `"auto"` to `"in-memory"` anyway.
    """
    engine = get_engine_affinity()
    return "in-memory" if engine == "auto" else engine
