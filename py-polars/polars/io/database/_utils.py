from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from polars._utils.various import parse_version
from polars.convert import from_arrow
from polars.dependencies import _PYARROW_AVAILABLE, import_optional
from polars.exceptions import ModuleUpgradeRequiredError

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from polars import DataFrame
    from polars._typing import SchemaDict


def _run_async(co: Coroutine[Any, Any, Any]) -> Any:
    """Run asynchronous code as if it was synchronous."""
    import asyncio

    import polars._utils.nest_asyncio

    polars._utils.nest_asyncio.apply()  # type: ignore[attr-defined]
    return asyncio.run(co)


def _read_sql_connectorx(
    query: str | list[str],
    connection_uri: str,
    partition_on: str | None = None,
    partition_range: tuple[int, int] | None = None,
    partition_num: int | None = None,
    protocol: str | None = None,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame:
    cx = import_optional("connectorx")
    try:
        return_type = "arrow2" if parse_version(cx.__version__) < (0, 4, 2) else "arrow"
        tbl = cx.read_sql(
            conn=connection_uri,
            query=query,
            return_type=return_type,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
        )
    except BaseException as err:
        # basic sanitisation of /user:pass/ credentials exposed in connectorx errs
        errmsg = re.sub("://[^:]+:[^:]+@", "://***:***@", str(err))
        raise type(err)(errmsg) from err

    return from_arrow(tbl, schema_overrides=schema_overrides)  # type: ignore[return-value]


def _read_sql_adbc(
    query: str,
    connection_uri: str,
    schema_overrides: SchemaDict | None,
    execute_options: dict[str, Any] | None = None,
) -> DataFrame:
    adbc_driver_manager = import_optional("adbc_driver_manager")
    adbc_str_version = getattr(adbc_driver_manager, "__version__", "0.0")
    adbc_version = parse_version(adbc_str_version)

    # From adbc_driver_manager version 1.6.0 Cursor.fetch_arrow() was introduced,
    # returning an object implementing the Arrow PyCapsule interface. This should be
    # used regardless of whether PyArrow is available.
    fetch_method_name = (
        "fetch_arrow" if adbc_version >= (1, 6, 0) else "fetch_arrow_table"
    )

    # From version 1.6.0 adbc_driver_manager no longer requires PyArrow for ordinary
    # (non-parameterised) queries.
    # From version 1.7.0 adbc_driver_manager no longer requires PyArrow for
    # passing Python sequences into parameterised queries (via execute_options)
    adbc_version_no_pyarrow_required = "1.6.0" if execute_options is None else "1.7.0"

    # Whether the user has the ADBC version they require to not have PyArrow
    has_required_adbc_version = adbc_version >= parse_version(
        adbc_version_no_pyarrow_required
    )

    if not has_required_adbc_version and not _PYARROW_AVAILABLE:
        msg_helper = (
            " when using parameterized queries (via `execute_options`)"
            if execute_options is not None
            else ""
        )
        msg = (
            f"pyarrow is required for adbc-driver-manager < "
            f"{adbc_version_no_pyarrow_required}{msg_helper}, found {adbc_str_version}.\n"
            "Either upgrade `adbc-driver-manager` (suggested) or install `pyarrow`"
        )
        raise ModuleUpgradeRequiredError(msg)
    with _open_adbc_connection(connection_uri) as conn, conn.cursor() as cursor:
        cursor.execute(query, **(execute_options or {}))
        tbl = getattr(cursor, fetch_method_name)()
        return from_arrow(tbl, schema_overrides=schema_overrides)  # type: ignore[return-value]


def _open_adbc_connection(connection_uri: str) -> Any:
    driver_name = connection_uri.split(":", 1)[0].lower()

    # map uri prefix to module when not 1:1
    module_suffix_map: dict[str, str] = {
        "postgres": "postgresql",
    }
    module_suffix = module_suffix_map.get(driver_name, driver_name)
    module_name = f"adbc_driver_{module_suffix}.dbapi"

    adbc_driver = import_optional(
        module_name,
        err_prefix="ADBC",
        err_suffix="driver not detected",
        install_message=f"If ADBC supports this database, please run: pip install adbc-driver-{driver_name}",
    )

    # some backends require the driver name to be stripped from the URI
    if driver_name in ("sqlite", "snowflake"):
        connection_uri = re.sub(f"^{driver_name}:/{{,3}}", "", connection_uri)

    return adbc_driver.connect(connection_uri)
