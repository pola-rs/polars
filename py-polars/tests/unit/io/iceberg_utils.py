from __future__ import annotations

import contextlib
import itertools
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic
import pyiceberg.table
import pytest

from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergScanResolver,
    IcebergScanTableSerializer,
    IcebergTableWrap,
)

if TYPE_CHECKING:
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table.sorting import SortOrder

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pydantic.warnings.PydanticDeprecatedSince212)
    warnings.simplefilter("ignore", DeprecationWarning)
    from pyiceberg.catalog.sql import SqlCatalog


@pytest.fixture(autouse=True)
def sqlcatalog_uses_null_pool_connections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # note: SqlCatalog is sqlite-backed, and pyiceberg builds its engine with
    # a SingletonThreadPool that can raise a ResourceWarning on test shutdown;
    # we ensure it uses a NullPool instead, resolving the connection "leak"
    import pyiceberg.catalog.sql as pyiceberg_sql
    from sqlalchemy.pool import NullPool

    tmp_path = tmp_path_factory.mktemp("iceberg_catalogs")
    original_create_engine = pyiceberg_sql.create_engine  # type: ignore[attr-defined]
    counter = itertools.count()

    def create_engine(url: str, **kwargs: Any) -> Any:
        if url == "sqlite:///:memory:":
            # need to use a file-backed catalog, otherwise NullPool connections
            # will be working with different in-memory databases on each call
            url = f"sqlite:///{tmp_path}/catalog_{next(counter)}.db"

        kwargs["poolclass"] = NullPool
        return original_create_engine(url, **kwargs)

    monkeypatch.setattr(pyiceberg_sql, "create_engine", create_engine)


def new_iceberg_scan_resolver(
    source: str | pyiceberg.table.Table,
) -> IcebergScanResolver:
    return IcebergScanResolver(
        table=IcebergTableWrap(
            table_=NoPickleOption(
                source if isinstance(source, pyiceberg.table.Table) else None
            ),
            table_descriptor_=source
            if not isinstance(source, pyiceberg.table.Table)
            else None,
            serializer=IcebergScanTableSerializer(),
            iceberg_storage_properties=None,
        ),
        snapshot_id=None,
        from_snapshot_id_exclusive=None,
        to_snapshot_id_inclusive=None,
        reader_override=None,
        use_metadata_statistics=True,
        fast_deletion_count=False,
        use_pyiceberg_filter=True,
    )


def new_iceberg_table(
    tmp_path: Path,
    *,
    schema: IcebergSchema,
    partition_spec: PartitionSpec | None = None,
    sort_order: SortOrder | None = None,
    name: str = "table",
    properties: dict[str, str] | None = None,
) -> tuple[pyiceberg.table.Table, SqlCatalog]:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    namespace = uuid.uuid4().bytes.hex()
    catalog.create_namespace(namespace)

    create_table_kwargs: dict[str, Any] = {"properties": properties or {}}
    if partition_spec is not None:
        create_table_kwargs["partition_spec"] = partition_spec
    if sort_order is not None:
        create_table_kwargs["sort_order"] = sort_order

    return catalog.create_table(
        (namespace, name), schema, **create_table_kwargs
    ), catalog


# PyIceberg on Windows uses `file://C:/` rather than `file:///C:/`.
def format_file_uri_iceberg(absolute_local_path: str | Path) -> str:
    absolute_local_path = str(absolute_local_path)

    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha()
        assert absolute_local_path[1] == ":"
        p = absolute_local_path.replace("\\", "/")
        return f"file://{p}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"


@pytest.fixture(name="iceberg_path")
def iceberg_path_fixture(io_files_path: Path) -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path(__file__).parent.resolve()

    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")  # noqa: PTH211

    iceberg_path = io_files_path / "iceberg-table" / "metadata" / "v2.metadata.json"
    return format_file_uri_iceberg(f"{iceberg_path.resolve()}")
