from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

import polars as pl
from polars._utils.logging import eprint
from polars._utils.various import parse_version
from polars._warnings import issue_warning
from polars.exceptions import ComputeError, PerformanceWarning, SchemaError
from polars.io.cloud.credential_provider._providers import (
    _get_credentials_from_provider_expiry_aware,
)
from polars.io.delta._utils import (
    _delta_cloud_root_is_ambiguous,
    _extract_table_statistics_from_delta_add_actions,
    _native_uri_from_delta_table_uri,
    _resolve_delta_lake_uri,
)
from polars.io.parquet.functions import scan_parquet
from polars.io.scan_options.cast_options import ScanCastOptions
from polars.lazyframe.resolver import LazyFrameResolver, ResolvedLazyFrameProps
from polars.schema import Schema

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Literal

    from deltalake import DeltaTable

    from polars._typing import DeletionFiles, StorageOptionsDict
    from polars.io.cloud._utils import NoPickleOption
    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder
    from polars.lazyframe.frame import LazyFrame
    from polars.lazyframe.resolver import FilterExpr


# Not from `deltalake.table`: those describe deltalake's own readers.
_MAX_READER_VERSION = 3
# Reader version 2 tables may use column mapping (v3 lists it as a feature), which is
# not supported: data files then store columns under physical names.
_UNSUPPORTED_READER_VERSION = 2
_SUPPORTED_FEATURES = frozenset(
    {
        "deletionVectors",
        "timestampNtz",
        "v2Checkpoint",
        "vacuumProtocolCheck",
        "variantType",
        "variantType-preview",
    }
)


def _check_reader_protocol(table: DeltaTable) -> None:
    """Raise if polars cannot read a table with this reader protocol."""
    from deltalake.exceptions import DeltaProtocolError

    table_protocol = table.protocol()
    min_reader_version = table_protocol.min_reader_version

    if min_reader_version == _UNSUPPORTED_READER_VERSION:
        msg = f"delta table requires reader version {min_reader_version} (column mapping), which polars does not support"
        raise DeltaProtocolError(msg)
    if min_reader_version > _MAX_READER_VERSION:
        msg = f"delta table requires reader version {min_reader_version}, polars supports up to {_MAX_READER_VERSION}"
        raise DeltaProtocolError(msg)
    if min_reader_version >= 3 and table_protocol.reader_features is not None:
        missing_features = {*table_protocol.reader_features}.difference(
            _SUPPORTED_FEATURES
        )
        if len(missing_features) > 0:
            msg = f"delta table requires reader features {sorted(missing_features)}, which polars does not support"
            raise DeltaProtocolError(msg)


def _table_id_of(table: DeltaTable) -> str | None:
    """The table's `metaData.id`, None if it has none (or deltalake reports none)."""
    table_id = table.metadata().id
    return str(table_id) if table_id else None


def _decode_binary_partition(value: str | bytes | None) -> bytes | None:
    """Bytes of a binary partition value, with Delta byte escapes decoded once."""
    if value is None:
        return None
    if isinstance(value, bytes):
        # deltalake returns the escape text as bytes.
        value = value.decode("latin-1")
    return re.sub(
        r"\\u00([0-9A-Fa-f]{2})",
        lambda m: chr(int(m[1], 16)),
        value,
    ).encode("latin-1")


def _table_root(table_uri: str) -> str:
    """Return the table prefix used by native scan paths."""
    # Match file_uris() for local paths and our lakefs-to-s3 rewrite.
    root = table_uri.removeprefix("file://").replace("lakefs://", "s3://", 1)
    # Windows: "file:///C:/t" -> "/C:/t", but file_uris() gives "C:/t".
    if len(root) > 2 and root[0] == "/" and root[2] == ":" and root[1].isalpha():
        root = root[1:]
    return root if root.endswith("/") else root + "/"


def _source_sizes(
    paths: list[str], root: str, sizes: dict[str, int]
) -> list[int] | None:
    """Return sizes in scan order using exact table-relative paths.

    Return None if any path is outside the root or has no logged size.
    """
    out = []
    for path in paths:
        if not path.startswith(root):
            return None
        size = sizes.get(path[len(root) :])
        if size is None:
            return None
        out.append(size)
    return out


def _metadata_row_count(
    statistics: pl.DataFrame | None,
    n_files: int,
    deletion_files: DeletionFiles | None,
    *,
    has_deletion_vectors: bool,
) -> tuple[int, int] | None:
    """Use complete physical counts and known native deletion cardinalities only."""
    if statistics is None or "len" not in statistics or statistics.height != n_files:
        return None
    counts = statistics["len"]
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    total = sum(counts)
    if total > (1 << 64) - 1:
        return None
    deleted = 0
    if has_deletion_vectors:
        if (
            deletion_files is None
            or deletion_files[0] != "delta-deletion-vector-native"
        ):
            # Compatibility callbacks do not expose metadata-only cardinalities.
            return None
        for index, vector in deletion_files[1].items():
            cardinality = vector[3]
            if (
                type(index) is not int
                or not 0 <= index < n_files
                or type(cardinality) is not int
                or not 0 <= cardinality <= counts[index]
            ):
                msg = "Delta deletion cardinality exceeds physical row count"
                raise ComputeError(msg)
            deleted += cardinality
    return total, deleted


@dataclass(kw_only=True)
class DeltaScanResolver(LazyFrameResolver):
    """Resolve Delta snapshots into native or PyArrow scans."""

    table_: NoPickleOption[DeltaTable]
    table_uri_: str | None
    version: int | str | datetime | None

    storage_options: StorageOptionsDict | None
    credential_provider_builder: CredentialProviderBuilder | None
    delta_table_options: dict[str, Any] | None

    use_pyarrow: bool
    pyarrow_options: dict[str, Any] | None

    _table_version_pinned: bool = field(default=False, init=False, repr=False)
    _table_uri_escaped: bool | None = field(default=None, init=False, repr=False)
    _schema: Schema | None = field(default=None, init=False, repr=False)
    # With `use_pyarrow`, `_schema` is that of the Arrow dataset as constructed
    # (the caller's `schema` option can reorder, retype, drop or add columns);
    # this is the table's own Delta schema, checked when the table advances.
    _table_schema: Schema | None = field(default=None, init=False, repr=False)
    _resolved_version: int | None = field(default=None, init=False)
    # The Delta table's `metaData.id` at first resolution (or the supplied object's):
    # a table replaced at the same URI and version number has another one.
    _resolved_table_id: str | None = field(default=None, init=False, repr=False)
    _plugin_enabled: bool | None = field(default=None, init=False)
    _root_planner: Literal["plugin", "deltalake"] | None = field(
        default=None, init=False, repr=False
    )
    _plugin_table_schema: str | None = field(default=None, init=False, repr=False)
    _plugin_metadata_cache: Any = field(default=None, init=False, repr=False)
    # With `use_pyarrow`: the last dataset scan and its table version, so that the
    # schema and the first resolution of a version build one dataset. Not pickled.
    _pyarrow_frame: tuple[int, LazyFrame] | None = field(
        default=None, init=False, repr=False
    )

    def _plugin_scan(self, **options: Any) -> tuple[LazyFrame | None, int] | None:
        from polars.io.delta._plugin import (
            IncompatiblePluginError,
            _warn_plugin_unusable,
            plugin_planner_required,
            plugin_scan,
            use_plugin_planner,
        )

        if self.use_pyarrow:
            return None
        if self._root_planner == "deltalake":
            if plugin_planner_required():
                msg = "delta: this saved cloud path must be reopened with deltalake"
                raise ComputeError(msg)
            return None
        if self._plugin_enabled is None:
            self._plugin_enabled = use_plugin_planner()
        if not self._plugin_enabled:
            return None
        if (
            self._root_planner is None
            and not plugin_planner_required()
            and self._caller_cloud_root_is_ambiguous()
        ):
            # The same spelling is a literal-percent key natively but a decoded
            # prefix for deltalake, which released Polars used: the default call
            # keeps that meaning. `POLARS_DELTA_PLANNER=plugin` opts into the literal
            # key; table() records the choice so that it never changes afterwards.
            self._require_deltalake_for_ambiguous_root()
            return None
        try:
            return plugin_scan(self, **options)
        except NotImplementedError as e:
            if plugin_planner_required():
                raise
            if self._root_planner == "plugin":
                # Native planning read this raw cloud key, which deltalake would
                # decode into another prefix: reading it would change the table.
                msg = (
                    f"delta: the cloud path {self._native_table_uri()!r} is ambiguous "
                    "(a percent escape, `?` or `#`): polars-delta opened the literal "
                    "key but cannot read that table, and deltalake would open a "
                    f"different, decoded one instead ({e}). Use the exact key "
                    "without such characters or a table that polars-delta supports."
                )
                raise ComputeError(msg) from e
            self._plugin_enabled = False
            return None
        except ValueError as e:
            if plugin_planner_required() or not self._supplied_snapshot_expired(e):
                raise
            # The caller's DeltaTable holds its loaded snapshot, but its version's
            # log file was removed (a checkpoint replaced it): the native planner
            # reopens the log by version, so plan from the object itself instead.
            self._plugin_enabled = False
            return None
        except IncompatiblePluginError:
            if plugin_planner_required():
                raise
            _warn_plugin_unusable("it is incompatible with this version of Polars")
            self._plugin_enabled = False
            return None

    def _expected_table_id(self) -> str | None:
        """The table identity a native expansion must match (None: not known yet)."""
        table = self.table_.get()
        if (
            table is not None
            and self.version is None
            and not self._table_version_pinned
        ):
            # A supplied object holds its loaded metadata in memory.
            return _table_id_of(table)
        return self._resolved_table_id

    def _require_deltalake_for_ambiguous_root(self) -> None:
        """Raise a clear error if deltalake, which this path routes to, is missing."""
        try:
            import deltalake  # noqa: F401
        except ImportError as e:
            msg = (
                f"delta: the cloud path {self._native_table_uri()!r} is ambiguous: "
                "it reads as a literal percent escape (or `?`, `#`) in the object key, "
                "or as the decoded prefix. By default it is read as released Polars "
                "did (the decoded prefix), which needs `deltalake`; install it, or "
                "set POLARS_DELTA_PLANNER=plugin to read the raw path natively"
            )
            raise ImportError(msg) from e

    def _caller_cloud_root_is_ambiguous(self) -> bool:
        self.table_uri()
        return not self._table_uri_escaped and _delta_cloud_root_is_ambiguous(
            self._native_table_uri()
        )

    def _supplied_snapshot_expired(self, error: ValueError) -> bool:
        """Whether native planning could not reopen a supplied DeltaTable's version."""
        return (
            self.version is None
            and self.table_.get() is not None
            and "No table version found" in str(error)
        )

    #
    # LazyFrameResolver interface functions
    #

    def schema(self) -> Schema:
        """Fetch the schema of the table."""
        if self._schema is None:
            if self.use_pyarrow:
                table = self.table()
                self._table_schema = Schema(table.schema())
                self._schema = self._pyarrow_scan(table).collect_schema()
                self._resolved_version = table.version()
            # Reads the log's protocol and metadata only, not its file actions.
            elif (scan := self._plugin_scan(schema_only=True)) is not None:
                frame, _ = scan
                assert frame is not None
                self._schema = frame.collect_schema()
            else:
                table = self.table()
                self._schema = Schema(table.schema())
                self._resolved_version = table.version()
        return self._schema

    def _pyarrow_scan(self, table: DeltaTable) -> LazyFrame:
        version = table.version()
        if self._pyarrow_frame is not None and self._pyarrow_frame[0] == version:
            return self._pyarrow_frame[1]
        frame = pl.scan_pyarrow_dataset(
            table.to_pyarrow_dataset(**(self.pyarrow_options or {}))
        )
        self._pyarrow_frame = (version, frame)
        return frame

    def cse_eq(self, other: object) -> bool:
        return self is other

    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],
        filter_columns: list[str],
        filter_drop_columns_idx: int | None,
        existing_resolved_version_key: str | None,
    ) -> tuple[LazyFrame | None, ResolvedLazyFrameProps]:
        """Return an unsliced scan; the consumer applies filters and projection."""
        # Statistics can only prune filters inside the resolved subplan. Join keys
        # and projected columns no longer need column statistics at this stage.
        del filters, filter_drop_columns_idx, limit
        if projection is not None:
            projection = list(dict.fromkeys([*projection, *filter_columns]))
        statistics_filter_columns = filter_columns or None
        frame, version_key = self._resolve_scan(
            existing_resolved_version_key=existing_resolved_version_key,
            projection=projection,
            filter_columns=statistics_filter_columns,
        )
        return frame, ResolvedLazyFrameProps(version_key=version_key)

    def _resolve_scan(
        self,
        *,
        existing_resolved_version_key: str | None,
        projection: list[str] | None,
        filter_columns: list[str] | None,
    ) -> tuple[LazyFrame | None, str]:
        import polars as pl
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "DeltaScanResolver: resolve_lazyframe(): "
                f"version: {self.version}, "
                f"projection: {projection}, "
                f"filter_columns: {filter_columns}, "
                f"use_pyarrow: {self.use_pyarrow}"
            )

        if (
            scan := self._plugin_scan(
                existing_resolved_version_key=existing_resolved_version_key,
                limit=None,
                projection=projection,
                filter_columns=filter_columns,
                statistics_columns=None,
            )
        ) is not None:
            frame, resolved_version = scan
            version_key = str(resolved_version)
            if existing_resolved_version_key == version_key:
                if verbose:
                    eprint(
                        f"DeltaScanResolver: resolve_lazyframe(): early return ({version_key = })"
                    )
                return None, version_key
            assert frame is not None
            return frame, version_key

        table = self.table()
        self._resolved_version = table.version()
        version_key = str(self._resolved_version)
        expected_table_schema = self._table_schema if self.use_pyarrow else self._schema
        if (
            expected_table_schema is not None
            and Schema(table.schema()) != expected_table_schema
        ):
            msg = "DeltaTable schema changed after scan creation; create a new scan"
            raise SchemaError(msg)

        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == version_key
        ):
            if verbose:
                eprint(
                    f"DeltaScanResolver: resolve_lazyframe(): early return ({version_key = })"
                )

            return None, version_key

        if self.use_pyarrow:
            frame = self._pyarrow_scan(table)
            if self._table_schema is None:
                # Saved before the Delta schema was kept apart: `_schema` may be the
                # Delta schema rather than the dataset's (a `schema` option changes
                # it), so it cannot be compared. Derive both again.
                self._table_schema = Schema(table.schema())
                self._schema = frame.collect_schema()
            elif self._schema is not None and frame.collect_schema() != self._schema:
                msg = "PyArrow dataset schema changed after scan creation; create a new scan"
                raise SchemaError(msg)
            return frame, version_key

        table_md = table.metadata()
        partition_columns = set(table_md.partition_columns)

        schema = self.schema()
        hive_schema = Schema(
            {k: v for k, v in schema.items() if k in partition_columns}
        )

        start_time = perf_counter()

        if verbose:
            eprint("DeltaScanResolver: resolve_lazyframe(): begin path expansion")

        paths = table.file_uris()

        if self.table_uri().startswith("lakefs://"):
            paths = [path.replace("lakefs://", "s3://") for path in paths]

        if verbose:
            elapsed = perf_counter() - start_time
            eprint(
                "DeltaScanResolver: resolve_lazyframe(): "
                f"native scan_parquet(): "
                f"num_files: {len(paths)}, "
                f"path expansion time: {elapsed:.3f}s"
            )

        # Private delta-rs API: fetch sizes without materializing add-action statistics.
        source_sizes = _source_sizes(
            paths, _table_root(table.table_uri), table._table.get_add_file_sizes()
        )

        if source_sizes is None and verbose:
            eprint(
                "DeltaScanResolver: resolve_lazyframe(): "
                "cannot pair add file sizes with file_uris(), skipping sizes"
            )

        # Row totals remain useful after resolution, including for runtime join
        # filters. Only filtered columns need their three column-statistics arrays.
        add_actions = pl.DataFrame(table.get_add_actions())
        if filter_columns is not None:
            table_statistics = _extract_table_statistics_from_delta_add_actions(
                add_actions,
                filter_columns=filter_columns,
                best_effort_columns=[],
                schema=schema,
                verbose=verbose,
            )
        elif "num_records" in add_actions:
            table_statistics = add_actions.select(pl.col("num_records").alias("len"))
        else:
            table_statistics = None
        hive_partitions = None
        if partition_columns:
            if not all(
                path.endswith(unquote(add_path))
                for path, add_path in zip(paths, add_actions["path"], strict=True)
            ):
                msg = "delta add actions are not in file_uris() order"
                raise ComputeError(msg)
            hive_partitions = add_actions.select(
                pl.col("partition").struct.field(*hive_schema)
            )
            # Binary partition values are `\uXXXX` escapes, one per byte.
            hive_partitions = hive_partitions.with_columns(
                pl.col(name).map_elements(
                    _decode_binary_partition, return_dtype=pl.Binary
                )
                for name, dtype in hive_schema.items()
                if dtype == pl.Binary
            ).cast(hive_schema)

        reader_features = table.protocol().reader_features
        has_deletion_vectors = (
            reader_features is not None and "deletionVectors" in reader_features
        )

        deletion_files: DeletionFiles | None = None
        decoder_available = False
        if has_deletion_vectors:
            from polars.io.delta._deletion_vectors import _native_deletion_vectors

            if add_actions is None:
                add_actions = pl.DataFrame(table.get_add_actions())
            from polars.io.delta._plugin import dv_decoder_available

            # Native descriptors are decoded by the `polars-delta` plugin.
            decoder_available = dv_decoder_available()
            if decoder_available:
                deletion_files = _native_deletion_vectors(
                    add_actions, paths, _table_root(table.table_uri)
                )
        if has_deletion_vectors and deletion_files is None:
            import deltalake

            dv_min_version = (1, 4, 2)
            installed = parse_version(deltalake.__version__)
            if installed < dv_min_version:
                msg = (
                    f"reading delta deletion vectors requires "
                    f"deltalake >= {'.'.join(str(v) for v in dv_min_version)}, "
                    f"found {installed}"
                )
                if not decoder_available:
                    msg += (
                        "; alternatively install polars-delta, which plans the "
                        "table itself"
                    )
                raise ImportError(msg + ".")
            if not decoder_available:
                from polars.io.delta._plugin import (
                    PLANNER_ENV_VAR,
                    _planner,
                    dv_decoder_supported_build,
                )

                if not dv_decoder_supported_build():
                    reason = (
                        "This Polars build cannot decode Delta deletion vectors "
                        "natively (it lacks cloud and parquet support)"
                    )
                else:
                    reason = "Install or upgrade polars-delta to read Delta deletion vectors lazily"
                if _planner() != "deltalake":
                    issue_warning(
                        f"{reason}. Loading them eagerly through "
                        "DeltaTable.deletion_vectors(); set "
                        f"{PLANNER_ENV_VAR}=deltalake to use deltalake without "
                        "this warning.",
                        PerformanceWarning,
                    )

            def _deletion_vector_callback(
                requested_paths: pl.DataFrame,
            ) -> pl.DataFrame:
                delta_deletion_vectors = _fetch_deletion_vectors(table)
                if delta_deletion_vectors is None:
                    return pl.DataFrame(
                        {"selection_vector": [None] * len(requested_paths)},
                        schema={"selection_vector": pl.List(pl.Boolean)},
                    )
                return _extract_delta_deletion_vectors(
                    requested_paths,
                    delta_deletion_vectors,
                    table_root=_table_root(table.table_uri),
                )

            deletion_files = (
                "delta-deletion-vector",
                _deletion_vector_callback,
            )

        row_count = _metadata_row_count(
            table_statistics,
            len(paths),
            deletion_files,
            has_deletion_vectors=has_deletion_vectors,
        )

        return scan_parquet(
            paths,
            schema=schema,
            hive_schema=hive_schema if len(partition_columns) > 0 else None,
            hive_partitioning=len(partition_columns) > 0,
            cast_options=ScanCastOptions._default_iceberg(),
            missing_columns="insert",
            extra_columns="ignore",
            storage_options=self.storage_options,
            credential_provider=self.credential_provider_builder,  # type: ignore[arg-type]
            _table_statistics=table_statistics,
            _row_count=row_count,
            _hive_partitions=hive_partitions,
            _deletion_files=deletion_files,
            _source_sizes=source_sizes,
        ), version_key

    #
    # Accessors
    #

    def table_uri(self) -> str:
        """Fetch the table URI."""
        if self._table_uri_escaped is None:
            if self.table_uri_ is None:
                table = self.table_.get()
                assert table is not None
                # Not via table(): pickling must not run the protocol check.
                self.table_uri_ = table.table_uri
                self._table_uri_escaped = True
            else:
                self.table_uri_ = _resolve_delta_lake_uri(self.table_uri_, strict=False)
                # Older pickles lack the URI flag. A stored plugin table schema
                # identifies a scan created from a DeltaTable without time travel.
                self._table_uri_escaped = self._plugin_table_schema is not None

        assert self.table_uri_ is not None
        return self.table_uri_

    def _native_table_uri(self) -> str:
        uri = self.table_uri()
        return _native_uri_from_delta_table_uri(uri) if self._table_uri_escaped else uri

    def table(self) -> DeltaTable:
        """Fetch the DeltaTable object."""
        if self._root_planner == "plugin":
            msg = (
                "delta: deltalake cannot reliably reopen this native Delta cloud "
                "path. Use the polars-delta planner to read this scan."
            )
            raise ComputeError(msg)
        table = self.table_.get()
        if (
            table is not None
            and self.version is not None
            and not self._table_version_pinned
        ):
            # Explicit time travel must not follow mutations of a caller-owned table.
            self.table_uri()
            table = None
        if table is None:
            from polars.io.delta._utils import _get_delta_lake_table

            credential_provider_creds = {}

            if self.credential_provider_builder and (
                provider := self.credential_provider_builder.build_credential_provider()
            ):
                credential_provider_creds = (
                    _get_credentials_from_provider_expiry_aware(provider) or {}
                )

            table = _get_delta_lake_table(
                table_path=self.table_uri(),
                version=self._resolved_version
                if self._resolved_version is not None
                else self.version,
                storage_options=(
                    {**(self.storage_options or {}), **credential_provider_creds}
                    if self.storage_options is not None
                    or self.credential_provider_builder is not None
                    else None
                ),
                delta_table_options=self.delta_table_options,
            )

            self.table_.set(table)
            self._table_version_pinned = True

        # Also checks a user-supplied DeltaTable.
        _check_reader_protocol(table)
        if not self.use_pyarrow and _delta_cloud_root_is_ambiguous(
            self._native_table_uri()
        ):
            self._root_planner = "deltalake"
        return table

    def __getstate__(self) -> dict[str, Any]:
        self.table_uri()
        state = self.__dict__.copy()
        table = self.table_.get()
        if (
            table is not None
            and self.version is None
            and not self._table_version_pinned
        ):
            # The caller's loaded object is dropped from the state: save the version
            # it holds (in memory, no I/O) so the restored scan reads exactly that
            # snapshot, not an older resolved one or the latest.
            state["_resolved_version"] = table.version()
            state["_resolved_table_id"] = _table_id_of(table)
        state["_plugin_metadata_cache"] = None
        state["_pyarrow_frame"] = None
        state["_plugin_enabled"] = None
        state["_root_planner"] = self._root_planner
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        if "_root_planner" not in state:
            native_guard = state.pop("_plugin_root_requires_native", None)
            if native_guard:
                self._root_planner = "plugin"
            elif state.pop("_legacy_root_requires_deltalake", False):
                self._root_planner = "deltalake"
            elif (
                self._resolved_version is not None
                and (uri := self.table_uri_) is not None
                and _delta_cloud_root_is_ambiguous(uri)
            ):
                self._recover_legacy_cloud_root(uri, native_guard=native_guard)
        self._plugin_enabled = None

    def _recover_legacy_cloud_root(
        self, uri: str, *, native_guard: bool | None
    ) -> None:
        plugin_enabled = self._plugin_enabled
        if self.use_pyarrow or plugin_enabled is False or uri.startswith("lakefs://"):
            self._root_planner = "deltalake"
            return

        native_uri = _native_uri_from_delta_table_uri(uri)
        escaped = self._table_uri_escaped
        if escaped is None and self._plugin_table_schema is not None:
            escaped = True
        # Native planning recorded the schema of a DeltaTable with no explicit
        # version. For time travel, old state may omit any sign of that origin.
        if escaped is None and plugin_enabled is True and self.version is None:
            escaped = False
        if escaped is None and native_uri != uri:
            msg = "delta: cannot recover the original Delta cloud path from this saved scan. Recreate the scan from its source."
            raise ComputeError(msg)

        if _delta_cloud_root_is_ambiguous(native_uri if escaped else uri):
            if native_guard is False:
                # This state was saved after native planning started recording
                # ambiguous roots. The absent restriction establishes deltalake use.
                self._root_planner = "deltalake"
            elif plugin_enabled is None:
                msg = "delta: cannot recover the original planner for this saved Delta cloud path. Recreate the scan from its source."
                raise ComputeError(msg)
            else:
                self._root_planner = "plugin"


# Characters delta-rs leaves unescaped in cloud file_uris(): printable ASCII except the
# URL path percent-encode set (space, `"`, `<`, `>`, backtick, `{` and `}`). Its URL
# join reads `#` and `?` as fragment and query delimiters, so these stay unescaped too.
_URL_PATH_SAFE = "!#$%&'()*+,-./:;=?@[\\]^_|~"


def _dv_path_key(path: str, table_root: str | None) -> str:
    """Return the deletion vector `path` as file_uris() gives it.

    Below the table root, file_uris() decodes local paths and gives cloud paths as a
    URL path, while deletion vector paths keep the log's percent-encoding.
    """
    root = table_root if table_root and path.startswith(table_root) else ""
    rel = unquote(path[len(root) :])
    if "://" in path:
        rel = quote(rel, safe=_URL_PATH_SAFE)
    return root + rel


def _extract_delta_deletion_vectors(
    requested_paths: pl.DataFrame,
    delta_deletion_vectors: pl.DataFrame,
    table_root: str | None = None,
) -> pl.DataFrame:
    """
    Extract the deletion_vectors for the provided requested_paths.

    Input requested_paths schema is "path": String.
    Output series schema is "selection_vector": List(Boolean), maintaining order.

    The selection_vector from deltalake is a keep-mask (True = keep).

    `table_root` is the table prefix of the requested paths (see `_table_root`).
    """
    assert requested_paths.schema == {"path": pl.String}

    delta_dv_schema = {"filepath": pl.String, "selection_vector": pl.List(pl.Boolean)}
    delta_deletion_vectors = delta_deletion_vectors.select(delta_dv_schema.keys())
    assert delta_deletion_vectors.schema == delta_dv_schema

    file_prefix = "file://" if sys.platform != "win32" else "file:///"
    delta_deletion_vectors = delta_deletion_vectors.with_columns(
        pl.col("filepath")
        .str.replace("^lakefs://", "s3://")
        .str.strip_prefix(file_prefix)
    )
    # Only paths with a percent-escape can differ from their file_uris() form.
    filepath = delta_deletion_vectors["filepath"]
    escaped = filepath.str.contains("%", literal=True).fill_null(False).arg_true()
    if len(escaped) > 0:
        keys = [_dv_path_key(p, table_root) for p in filepath.gather(escaped).to_list()]
        delta_deletion_vectors = delta_deletion_vectors.with_columns(
            filepath.clone().scatter(escaped, keys)
        )
    joined_df = (
        requested_paths.lazy()
        .with_columns(
            pl.col("path")
            .str.replace("^lakefs://", "s3://")
            .str.strip_prefix(file_prefix)
        )
        .join(
            delta_deletion_vectors.lazy(),
            left_on="path",
            right_on="filepath",
            how="left",
            maintain_order="left",
        )
        .select(["selection_vector"])
        ._collect_eager()
    )

    assert joined_df.height == len(requested_paths)

    return joined_df


def _fetch_deletion_vectors(table: DeltaTable) -> pl.DataFrame | None:
    """
    Fetch the deletion_vectors, mapping file_uri to "deletion_vector".

    Schema: {"filepath": pl.String, "selection_vector": pl.List(pl.Boolean)}

    The selection_vector from deltalake is a keep-mask (True = keep), so
    the more accurate term would be "selection_vector".

    Returns None if the table has no deletion vectors.
    """
    import polars._utils.logging

    verbose = polars._utils.logging.verbose()

    dv_table = pl.DataFrame(table.deletion_vectors())

    if verbose and dv_table.height > 0:
        eprint(f"DeltaScanResolver: has deletion_vectors, file_count: {len(dv_table)}")

    if len(dv_table) == 0:
        return None

    return dv_table
