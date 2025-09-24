from __future__ import annotations

import re
from importlib import import_module
from typing import TYPE_CHECKING, Any

from polars._dependencies import import_optional
from polars._utils.various import parse_version
from polars.convert import from_arrow
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
    pre_execution_query: str | list[str] | None = None,
) -> DataFrame:
    cx = import_optional("connectorx")

    if parse_version(cx.__version__) < (0, 4, 2):
        if pre_execution_query:
            msg = "'pre_execution_query' is only supported in connectorx version 0.4.2 or later"
            raise ValueError(msg)
        return_type = "arrow2"
        pre_execution_args = {}
    else:
        return_type = "arrow"
        pre_execution_args = {"pre_execution_query": pre_execution_query}

    try:
        tbl = cx.read_sql(
            conn=connection_uri,
            query=query,
            return_type=return_type,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
            **pre_execution_args,
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
    with _open_adbc_connection(connection_uri) as conn, conn.cursor() as cursor:
        cursor.execute(query, **(execute_options or {}))
        tbl = cursor.fetch_arrow_table()
    return from_arrow(tbl, schema_overrides=schema_overrides)  # type: ignore[return-value]


def _get_adbc_driver_name_from_uri(connection_uri: str) -> str:
    driver_name = connection_uri.split(":", 1)[0].lower()
    # map uri prefix to ADBC name when not 1:1
    driver_suffix_map: dict[str, str] = {"postgres": "postgresql"}
    return driver_suffix_map.get(driver_name, driver_name)


def _get_adbc_module_name_from_uri(connection_uri: str) -> str:
    driver_name = _get_adbc_driver_name_from_uri(connection_uri)
    return f"adbc_driver_{driver_name}"


def _import_optional_adbc_driver(
    module_name: str,
    *,
    dbapi_submodule: bool = True,
) -> Any:
    # Always import top level module first. This will surface a better error for users
    # if the module does not exist. It doesn't negatively impact performance given the
    # dbapi submodule would also load it.
    adbc_driver = import_optional(
        module_name,
        err_prefix="ADBC",
        err_suffix="driver not detected",
        install_message=(
            "If ADBC supports this database, please run: pip install "
            f"{module_name.replace('_', '-')}"
        ),
    )
    if not dbapi_submodule:
        return adbc_driver
    # Importing the dbapi without pyarrow before adbc_driver_manager 1.6.0
    # raises ImportError: PyArrow is required for the DBAPI-compatible interface
    # Use importlib.import_module because Polars' import_optional clobbers this error
    try:
        adbc_driver_dbapi = import_module(f"{module_name}.dbapi")
    except ImportError as e:
        if "PyArrow is required for the DBAPI-compatible interface" in (str(e)):
            adbc_driver_manager = import_optional("adbc_driver_manager")
            adbc_str_version = getattr(adbc_driver_manager, "__version__", "0.0")

            msg = (
                "pyarrow is required for adbc-driver-manager < 1.6.0, found "
                f"{adbc_str_version}.\nEither upgrade `adbc-driver-manager` (suggested) or "
                "install `pyarrow`"
            )
            raise ModuleUpgradeRequiredError(msg) from None
        # if the error message was something different, re-raise it
        raise
    else:
        return adbc_driver_dbapi


def _open_adbc_connection(connection_uri: str) -> Any:
    driver_name = _get_adbc_driver_name_from_uri(connection_uri)
    module_name = _get_adbc_module_name_from_uri(connection_uri)
    adbc_driver = _import_optional_adbc_driver(module_name)

    # some backends require the driver name to be stripped from the URI
    if driver_name in ("duckdb", "snowflake", "sqlite"):
        connection_uri = re.sub(f"^{driver_name}:/{{,3}}", "", connection_uri)

    return adbc_driver.connect(connection_uri)
