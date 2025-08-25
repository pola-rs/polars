from __future__ import annotations

import os
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import polars._reexport as pl
from polars._utils.logging import eprint, verbose
from polars.exceptions import ComputeError
from polars.io.iceberg._utils import (
    IdentityTransformedPartitionValuesBuilder,
    _scan_pyarrow_dataset_impl,
)
from polars.io.scan_options.cast_options import ScanCastOptions

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
        reader_override: Literal["native", "pyiceberg"] | None = None,
    ) -> None:
        self._metadata_path = None
        self._table = None
        self._snapshot_id = snapshot_id
        self._iceberg_storage_properties = iceberg_storage_properties
        self._reader_override: Literal["native", "pyiceberg"] | None = reader_override

        # Accept either a path or a table object. The one we don't have is
        # lazily initialized when needed.

        if isinstance(source, str):
            self._metadata_path = source
        else:
            self._table = source

    #
    # PythonDatasetProvider interface functions
    #

    def schema(self) -> pa.schema:
        """Fetch the schema of the table."""
        return self.arrow_schema()

    def arrow_schema(self) -> pa.schema:
        """Fetch the arrow schema of the table."""
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        return schema_to_pyarrow(self.table().schema())

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
    ) -> tuple[LazyFrame, str] | None:
        """Construct a LazyFrame scan."""
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "IcebergDataset: to_dataset_scan(): "
                f"snapshot ID: {self._snapshot_id}, "
                f"limit: {limit}, "
                f"projection: {projection}"
            )

        tbl = self.table()

        if verbose:
            eprint(
                "IcebergDataset: to_dataset_scan(): "
                f"tbl.metadata.current_snapshot_id: {tbl.metadata.current_snapshot_id}"
            )

        selected_fields = ("*",) if projection is None else tuple(projection)

        snapshot_id = self._snapshot_id

        def snapshot_id_not_found(snapshot_id: Any) -> ValueError:
            return ValueError(f"iceberg snapshot ID not found: {snapshot_id}")

        # Take from parameter first then envvar
        reader_override = self._reader_override or os.getenv(
            "POLARS_ICEBERG_READER_OVERRIDE"
        )

        if reader_override and reader_override not in ["native", "pyiceberg"]:
            msg = (
                "iceberg: unknown value for reader_override: "
                f"'{reader_override}', expected one of ('native', 'pyiceberg')"
            )
            raise ValueError(msg)

        fallback_reason = (
            "forced reader_override='pyiceberg'"
            if reader_override == "pyiceberg"
            else f"unsupported table format version: {tbl.format_version}"
            if not tbl.format_version <= 2
            else None
        )

        schema_id = None

        if snapshot_id is not None:
            snapshot = tbl.snapshot_by_id(snapshot_id)

            if snapshot is None:
                raise snapshot_id_not_found(snapshot_id)

            schema_id = snapshot.schema_id

            if schema_id is None:
                msg = (
                    f"IcebergDataset: requested snapshot {snapshot_id} "
                    "did not contain a schema ID"
                )
                raise ValueError(msg)

            iceberg_schema = tbl.schemas()[schema_id]
            snapshot_id_key = f"{snapshot.snapshot_id}"
        else:
            iceberg_schema = tbl.schema()
            schema_id = tbl.metadata.current_schema_id

            snapshot_id_key = (
                f"{v.snapshot_id}" if (v := tbl.current_snapshot()) is not None else ""
            )

        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == snapshot_id_key
        ):
            if verbose:
                eprint(
                    "IcebergDataset: to_dataset_scan(): early return "
                    f"({snapshot_id_key = })"
                )

            return None

        projected_iceberg_schema = (
            iceberg_schema
            if selected_fields == ("*",)
            else iceberg_schema.select(*selected_fields)
        )

        sources = []
        missing_field_defaults = IdentityTransformedPartitionValuesBuilder(
            tbl,
            projected_iceberg_schema,
        )
        deletion_files: dict[int, list[str]] = {}

        if reader_override != "pyiceberg" and not fallback_reason:
            from pyiceberg.manifest import DataFileContent, FileFormat

            if verbose:
                eprint("IcebergDataset: to_dataset_scan(): begin path expansion")

            start_time = perf_counter()

            scan = tbl.scan(
                snapshot_id=snapshot_id,
                limit=limit,
                selected_fields=selected_fields,
            )

            total_deletion_files = 0

            for i, file_info in enumerate(scan.plan_files()):
                if file_info.file.file_format != FileFormat.PARQUET:
                    fallback_reason = (
                        f"non-parquet format: {file_info.file.file_format}"
                    )
                    break

                if file_info.delete_files:
                    deletion_files[i] = []

                    for deletion_file in file_info.delete_files:
                        if deletion_file.content != DataFileContent.POSITION_DELETES:
                            fallback_reason = (
                                "unsupported deletion file type: "
                                f"{deletion_file.content}"
                            )
                            break

                        if deletion_file.file_format != FileFormat.PARQUET:
                            fallback_reason = (
                                "unsupported deletion file format: "
                                f"{deletion_file.file_format}"
                            )
                            break

                        deletion_files[i].append(deletion_file.file_path)
                        total_deletion_files += 1

                if fallback_reason:
                    break

                missing_field_defaults.push_partition_values(
                    current_index=i,
                    partition_spec_id=file_info.file.spec_id,
                    partition_values=file_info.file.partition,
                )

                sources.append(file_info.file.file_path)

            if verbose:
                elapsed = perf_counter() - start_time
                eprint(
                    "IcebergDataset: to_dataset_scan(): "
                    f"finish path expansion ({elapsed:.3f}s)"
                )

        if not fallback_reason:
            if verbose:
                s = "" if len(sources) == 1 else "s"
                s2 = "" if total_deletion_files == 1 else "s"

                eprint(
                    "IcebergDataset: to_dataset_scan(): "
                    f"native scan_parquet(): "
                    f"{len(sources)} source{s}, "
                    f"snapshot ID: {snapshot_id}, "
                    f"schema ID: {schema_id}, "
                    f"{total_deletion_files} deletion file{s2}"
                )

            from polars.io.parquet.functions import scan_parquet

            return scan_parquet(
                sources,
                cast_options=ScanCastOptions._default_iceberg(),
                missing_columns="insert",
                extra_columns="ignore",
                _column_mapping=(
                    "iceberg-column-mapping",
                    # The arrow schema returned by `schema_to_pyarrow` will contain
                    # 'PARQUET:field_id'
                    schema_to_pyarrow(iceberg_schema),
                ),
                _default_values=("iceberg", missing_field_defaults.finish()),
                _deletion_files=("iceberg-position-delete", deletion_files),
            ), snapshot_id_key

        elif reader_override == "native":
            msg = f"iceberg reader_override='native' failed: {fallback_reason}"
            raise ComputeError(msg)

        if verbose:
            eprint(
                "IcebergDataset: to_dataset_scan(): "
                f"fallback to python[pyiceberg] scan: {fallback_reason}"
            )

        func = partial(
            _scan_pyarrow_dataset_impl,
            tbl,
            snapshot_id=snapshot_id,
            n_rows=limit,
            with_columns=projection,
        )

        arrow_schema = schema_to_pyarrow(tbl.schema())

        lf = pl.LazyFrame._scan_python_function(arrow_schema, func, pyarrow=True)

        return lf, snapshot_id_key

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
            "reader_override": self._reader_override,
        }

        if verbose():
            path_repr = state["metadata_path"]
            snapshot_id = f"'{v}'" if (v := state["snapshot_id"]) is not None else None
            keys_repr = _redact_dict_values(state["iceberg_storage_properties"])
            reader_override = state["reader_override"]

            eprint(
                "IcebergDataset: getstate(): "
                f"path: '{path_repr}', "
                f"snapshot_id: {snapshot_id}, "
                f"iceberg_storage_properties: {keys_repr}, "
                f"reader_override: {reader_override}"
            )

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if verbose():
            path_repr = state["metadata_path"]
            snapshot_id = state["snapshot_id"]
            keys_repr = _redact_dict_values(state["iceberg_storage_properties"])
            reader_override = state["reader_override"]

            eprint(
                "IcebergDataset: getstate(): "
                f"path: '{path_repr}', "
                f"snapshot_id: '{snapshot_id}', "
                f"iceberg_storage_properties: {keys_repr}, "
                f"reader_override: {reader_override}"
            )

        IcebergDataset.__init__(
            self,
            state["metadata_path"],
            snapshot_id=state["snapshot_id"],
            iceberg_storage_properties=state["iceberg_storage_properties"],
            reader_override=state["reader_override"],
        )


def _redact_dict_values(obj: Any) -> Any:
    return (
        {k: "REDACTED" for k in obj.keys()}  # noqa: SIM118
        if isinstance(obj, dict)
        else f"<{type(obj).__name__} object>"
        if obj is not None
        else "None"
    )
