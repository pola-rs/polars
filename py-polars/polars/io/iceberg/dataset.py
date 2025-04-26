from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import polars._reexport as pl
from polars._utils.logging import eprint, verbose
from polars.exceptions import ComputeError
from polars.io.iceberg._utils import _scan_pyarrow_dataset_impl

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.table import Table

    from polars.lazyframe.frame import LazyFrame


class IcebergDataset:
    """Dataset interface for PyIceberg."""

    def __init__(
        self,
        source: str | Table,
        *,
        snapshot_id: int | None = None,
        iceberg_storage_properties: dict[str, Any] | None = None,
        force_scan_dispatch: Literal["native", "python"] | None = None,
    ) -> None:
        self._metadata_path = None
        self._table = None
        self._snapshot_id = snapshot_id
        self._iceberg_storage_properties = iceberg_storage_properties
        self._force_scan_dispatch: Literal["native", "python"] | None = (
            force_scan_dispatch
        )
        self._force_scan_dispatch_envvar = os.getenv("POLARS_ICEBERG_SCAN_DISPATCH")

        # Accept either a path or a table object. The one we don't have is
        # lazily initialized when needed.

        if isinstance(source, str):
            self._metadata_path = source
        else:
            self._table = source

    #
    # PythonDatasetProvider interface functions
    #

    def reader_name(self) -> str:
        """Name of the reader."""
        return "iceberg"

    def schema(self) -> pa.schema:
        """Fetch the schema of the table."""
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        return schema_to_pyarrow(self.table().schema())

    def to_dataset_scan(
        self,
        *,
        limit: int | None = None,
        projection: list[str] | None = None,
    ) -> LazyFrame:
        """Construct a LazyFrame scan."""
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "IcebergDataset: to_dataset_scan(): "
                f"limit: {limit}, "
                f"projection: {projection}"
            )

        tbl = self.table()

        selected_fields = ("*",) if projection is None else tuple(projection)

        snapshot_id = self._snapshot_id

        if snapshot_id is not None:
            if tbl.snapshot_by_id(snapshot_id) is None:
                msg = f"iceberg snapshot ID not found: {snapshot_id}"
                raise ValueError(msg)

        # Take from parameter first then envvar
        force_scan_dispatch = (
            self._force_scan_dispatch or self._force_scan_dispatch_envvar
        )

        if force_scan_dispatch and force_scan_dispatch not in ["native", "python"]:
            msg = (
                "iceberg: unknown value for force_scan_dispatch: "
                f"'{force_scan_dispatch}', expected one of ('native', 'python')"
            )
            raise ValueError(msg)

        # Try native scan
        fallback_reason = (
            "forced force_scan_dispatch='python'"
            if force_scan_dispatch == "python"
            # TODO: Enable native scans by default after we have type casting support,
            # currently it may fail if the dataset has changed types.
            else "native scans disabled by default"
            if force_scan_dispatch != "native"
            else None
        )

        sources = []

        if force_scan_dispatch != "python":
            scan = tbl.scan(
                snapshot_id=snapshot_id, limit=limit, selected_fields=selected_fields
            )

            for file_info in scan.plan_files():
                if file_info.file.file_format != "PARQUET":
                    fallback_reason = (
                        f"non-parquet format: {file_info.file.file_format}"
                    )
                elif file_info.delete_files:
                    fallback_reason = "unimplemented: dataset contained delete files"
                else:
                    sources.append(file_info.file.file_path)
                    continue

                break

        if not fallback_reason:
            from polars.io.parquet.functions import scan_parquet

            if verbose:
                eprint(
                    "IcebergDataset: to_dataset_scan(): "
                    f"native scan_parquet() ({len(sources)} sources)"
                )

            return scan_parquet(
                sources,
                allow_missing_columns=True,
            )

        elif force_scan_dispatch == "native":
            msg = f"iceberg force_scan_dispatch='native' failed: {fallback_reason}"
            raise ComputeError(msg)

        if verbose:
            eprint(
                f"IcebergDataset: to_dataset_scan(): fallback to python scan: "
                f"{fallback_reason}"
            )

        func = partial(
            _scan_pyarrow_dataset_impl,
            tbl,
            snapshot_id=snapshot_id,
            n_rows=limit,
            with_columns=projection,
        )

        from pyiceberg.io.pyarrow import schema_to_pyarrow

        arrow_schema = schema_to_pyarrow(tbl.schema())

        lf = pl.LazyFrame._scan_python_function(arrow_schema, func, pyarrow=True)

        return lf

    #
    # Accessors
    #

    def metadata_path(self) -> str:
        """Fetch the metadata path."""
        if self._metadata_path is None:
            if self._table is None:
                msg = "impl error: both metadata_path and table are None"
                raise ValueError(msg)

            self._metadata_path = self.table().metadata_location

        return self._metadata_path

    def table(self) -> Table:
        """Fetch the PyIceberg Table object."""
        if self._table is None:
            if self._metadata_path is None:
                msg = "impl error: both metadata_path and table are None"
                raise ValueError(msg)

            if verbose():
                eprint(f"IcebergDataset: construct table from {self._metadata_path = }")

            from pyiceberg.table import StaticTable

            self._table = StaticTable.from_metadata(
                metadata_location=self._metadata_path,
                properties=self._iceberg_storage_properties or {},
            )

        return self._table

    #
    # Serialization functions
    #
    # We don't serialize the iceberg table object - the remote machine should
    # use their own permissions to reconstruct the table object from the path.
    #

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "metadata_path": self.metadata_path(),
            "snapshot_id": self._snapshot_id,
            "iceberg_storage_properties": self._iceberg_storage_properties,
            "force_scan_dispatch": self._force_scan_dispatch,
        }

        if verbose():
            path_repr = state["metadata_path"]
            snapshot_id = state["snapshot_id"]
            keys_repr = _redact_dict_values(state["iceberg_storage_properties"])
            force_scan_dispatch = state["force_scan_dispatch"]

            eprint(
                "IcebergDataset: getstate(): "
                f"path: '{path_repr}', "
                f"snapshot_id: '{snapshot_id}', "
                f"iceberg_storage_properties: {keys_repr}, "
                f"force_scan_dispatch: {force_scan_dispatch}"
            )

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if verbose():
            path_repr = state["metadata_path"]
            snapshot_id = state["snapshot_id"]
            keys_repr = _redact_dict_values(state["iceberg_storage_properties"])
            force_scan_dispatch = state["force_scan_dispatch"]

            eprint(
                "IcebergDataset: getstate(): "
                f"path: '{path_repr}', "
                f"snapshot_id: '{snapshot_id}', "
                f"iceberg_storage_properties: {keys_repr}, "
                f"force_scan_dispatch: {force_scan_dispatch}"
            )

        IcebergDataset.__init__(
            self,
            state["metadata_path"],
            snapshot_id=state["snapshot_id"],
            iceberg_storage_properties=state["iceberg_storage_properties"],
            force_scan_dispatch=state["force_scan_dispatch"],
        )


def _redact_dict_values(obj: Any) -> Any:
    return (
        {k: "REDACTED" for k in obj.keys()}  # noqa: SIM118
        if isinstance(obj, dict)
        else f"<{type(obj).__name__} object>"
        if obj is not None
        else "None"
    )
