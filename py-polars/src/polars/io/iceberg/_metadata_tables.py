"""Read Iceberg metadata tables in SQL, as in `SELECT * FROM lake.ns.t$snapshots`."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from polars._utils.logging import eprint, verbose
from polars.exceptions import InvalidOperationError
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._catalog_resolver import catalog_from
from polars.io.iceberg._dataset import _load_catalog_table, _reusable_catalog
from polars.lazyframe.resolver._resolver import LazyFrameResolver

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyiceberg.catalog
    import pyiceberg.table

    import polars as pl
    from polars._typing import SchemaDict
    from polars.io.iceberg._dataset import IcebergCatalogConfig
    from polars.lazyframe.resolver._resolver import (
        FilterExpr,
        ResolvedLazyFrameProps,
    )

# The tables of PyIceberg's `Table.inspect`.
METADATA_TABLES = frozenset(
    {
        "all_data_files",
        "all_delete_files",
        "all_files",
        "all_manifests",
        "data_files",
        "delete_files",
        "entries",
        "files",
        "history",
        "manifests",
        "metadata_log_entries",
        "partitions",
        "refs",
        "snapshots",
    }
)

# PyIceberg needs a snapshot to read these.
_NEED_SNAPSHOT = frozenset({"entries", "partitions"})


def metadata_table_of(
    identifier: Sequence[str],
) -> tuple[tuple[str, ...], str] | None:
    """
    Return the table and the metadata table that `identifier` names, or None.

    The last part must be `<table>$<name>`, split at the last `$`, with a known name
    (ignoring case) and a non-empty table.
    """
    table, sep, name = identifier[-1].rpartition("$")
    name = name.lower()
    if not sep or not table or name not in METADATA_TABLES:
        return None
    return (*identifier[:-1], table), name


@dataclass(kw_only=True)
class IcebergMetadataTableResolver(LazyFrameResolver):
    """
    Resolve a metadata table of a catalog table, such as `ns.t$snapshots`.

    Loads once, at first use, like a catalog table scan.
    """

    table_identifier: tuple[str, ...]
    metadata_table: str
    # Either the catalog properties or the unresolved catalog name
    catalog_config: IcebergCatalogConfig | str
    catalog_: NoPickleOption[pyiceberg.catalog.Catalog]
    display_name: str
    # Loaded once, shared by schema() and resolve_lazyframe()
    resolved_: NoPickleOption[pl.LazyFrame] = field(
        default_factory=lambda: NoPickleOption(None)
    )
    lock_: threading.Lock = field(default_factory=threading.Lock, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["lock_"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock_ = threading.Lock()

    def schema(self) -> SchemaDict:
        """Fetch the schema of the metadata table."""
        return self._resolve().collect_schema()

    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],  # noqa: ARG002
        filter_columns: list[str],  # noqa: ARG002
        filter_drop_columns_idx: int | None,  # noqa: ARG002
        existing_resolved_version_key: str | None,  # noqa: ARG002
    ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        """Return the metadata table."""
        lf = self._resolve()
        # An early cut. The caller still applies these and the filters.
        if limit is not None:
            lf = lf.head(limit)
        if projection is not None:
            lf = lf.select(projection) if projection else lf.drop("*")
        return lf

    def cse_eq(self, other: object) -> bool:
        # Only itself: another resolver may have loaded another snapshot.
        return other is self

    def _resolve(self) -> pl.LazyFrame:
        """Return the frame, loading it at the first call."""
        with self.lock_:
            if (lf := self.resolved_.get()) is None:
                lf = self._load()
                self.resolved_.set(lf)
            return lf

    def _load(self) -> pl.LazyFrame:
        """Load the table and read its metadata table."""
        from polars.io.iceberg._dataset import IcebergCatalogTableDescriptor

        catalog = self.catalog_.get()
        if catalog is None:
            catalog = catalog_from(self.catalog_config)
        descriptor = IcebergCatalogTableDescriptor(
            table_identifier=self.table_identifier,
            catalog_config=self.catalog_config,
            catalog_=NoPickleOption(None),
            display_name=self.display_name,
        )
        table = _load_catalog_table(catalog, descriptor)
        if verbose():
            eprint(
                f"IcebergMetadataTableResolver: {self.display_name}: "
                f"metadata table {self.metadata_table!r}"
            )
        return _metadata_frame(table, self.metadata_table, self.display_name)


def _metadata_frame(
    table: pyiceberg.table.Table, name: str, display_name: str
) -> pl.LazyFrame:
    import polars as pl

    if name in _NEED_SNAPSHOT and table.current_snapshot() is None:
        msg = (
            f"metadata table '{display_name}' cannot be read: the table has no "
            f"snapshot, and `{name}` needs one"
        )
        raise InvalidOperationError(msg)
    arrow: Any = getattr(table.inspect, name)()
    return pl.from_arrow(arrow).lazy()  # type: ignore[union-attr]


def scan_catalog_metadata_table(
    table_identifier: tuple[str, ...],
    metadata_table: str,
    catalog_config: IcebergCatalogConfig | str,
    catalog_instance: pyiceberg.catalog.Catalog | None,
    *,
    display_name: str,
) -> pl.LazyFrame:
    """Lazily scan a metadata table of a catalog table, for SQL catalogs."""
    return IcebergMetadataTableResolver(
        table_identifier=table_identifier,
        metadata_table=metadata_table,
        catalog_config=catalog_config,
        catalog_=NoPickleOption(_reusable_catalog(catalog_instance)),
        display_name=display_name,
    ).lazy()
