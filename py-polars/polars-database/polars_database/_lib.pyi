# Type stub of the `polars_database._lib` extension module (`src/python/mod.rs`).
from typing import Any

__version__: str
_polars_io_plugin_ids: tuple[str, ...]

def _capsule(id: str) -> Any: ...
def _at_exit() -> None: ...
def scan_resolver(
    source: str,
    source_type: str | None,
    connection: Any,
    connection_kind: str,
    dialect: str | None,
    driver: str | None,
    connect_options: list[tuple[str, str]],
    execute_options: list[tuple[str, str]],
    python_options: Any,
    credential_provider: Any,
    params: Any,
    schema: Any,
    schema_overrides: Any,
    partition_on: str | None,
    partition_auto: bool,
    partition_num: int | None,
    partition_range: Any,
    partition_predicates: list[str] | None,
    order_by: list[str],
    consistency: str,
    max_connections: int | None,
    batch_size: int | None,
    pushdown: bool,
) -> Any: ...
def driver_statuses(
    dialect: str | None = None,
) -> list[tuple[str, str, bool, str | None, str | None, str]]: ...
def dialect_from_scheme(scheme: str) -> str: ...
def find_adbc_driver(name: str) -> tuple[str, str | None] | None: ...
def userinfo_end(rest: str, secret: bool = False) -> int | None: ...
def connection_secrets(
    target: str,
    kind: str,
    password: str | None,
    options: list[tuple[str, str]],
) -> list[str]: ...
def scrub(message: str, secrets: list[str]) -> str: ...
def file_target(uri: str) -> tuple[str, list[tuple[str, str]]]: ...
def adbc_db_kwargs(driver: str, uri: str) -> list[tuple[str, str]] | None: ...
def default_max_connections(dialect: str) -> int: ...
