from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError

pytest.importorskip("pyiceberg.catalog.sql")

import pyiceberg.catalog
from pyiceberg.catalog.noop import NoopCatalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.utils.config import Config

from polars.io.iceberg._catalog_resolver import check_catalog_name, resolve_catalog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pyiceberg.catalog import Catalog


@pytest.fixture(autouse=True)
def _reset_resolver() -> Iterator[None]:
    yield
    pl.Config.set_iceberg_catalog_resolver(None)


@pytest.fixture(autouse=True)
def sqlcatalog_uses_null_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    # the default pool keeps sqlite connections open past the test (ResourceWarning)
    import pyiceberg.catalog.sql as pyiceberg_sql
    from sqlalchemy.pool import NullPool

    create_engine = pyiceberg_sql.create_engine  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pyiceberg_sql,
        "create_engine",
        lambda url, **kwargs: create_engine(url, **{**kwargs, "poolclass": NullPool}),
    )


def new_catalog(tmp_path: Path, name: str, df: pl.DataFrame) -> SqlCatalog:
    path = tmp_path / name
    path.mkdir()
    catalog = SqlCatalog(
        name, uri=f"sqlite:///{path / 'catalog.db'}", warehouse=f"file://{path}"
    )
    catalog.create_namespace("ns")
    table = catalog.create_table(("ns", "t"), schema=df.to_arrow().schema)
    df.write_iceberg(table, mode="append")
    return catalog


@pytest.fixture
def sales(tmp_path: Path) -> SqlCatalog:
    return new_catalog(tmp_path, "sales", pl.DataFrame({"a": [1, 2]}))


def recording(catalog: Catalog) -> tuple[Callable[[str], Catalog], list[str]]:
    names: list[str] = []

    def resolve(name: str) -> Catalog:
        names.append(name)
        return catalog

    return resolve, names


def fail_resolver(name: str) -> Catalog:
    pytest.fail(f"resolver called for {name!r}")


def test_set_resolver_returns_config() -> None:
    assert pl.Config.set_iceberg_catalog_resolver(fail_resolver) is pl.Config


def test_resolve_catalog_uses_installed_resolver(sales: SqlCatalog) -> None:
    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)

    assert resolve_catalog("sales") is sales
    assert names == ["sales"]


def test_resolver_none_restores_default(
    sales: SqlCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    def load_catalog(name: str, **_: Any) -> Catalog:
        loaded.append(name)
        return sales

    monkeypatch.setattr(pyiceberg.catalog, "load_catalog", load_catalog)
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)
    pl.Config.set_iceberg_catalog_resolver(None)

    assert resolve_catalog("sales") is sales
    assert loaded == ["sales"]


def test_config_context_restores_resolver(sales: SqlCatalog) -> None:
    previous, previous_names = recording(sales)
    inner, inner_names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(previous)

    with pl.Config(set_iceberg_catalog_resolver=inner):
        resolve_catalog("inner")
    resolve_catalog("outer")

    assert inner_names == ["inner"]
    assert previous_names == ["outer"]


def test_default_resolver_reads_pyiceberg_env(
    sales: SqlCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYICEBERG_CATALOG__SALES__TYPE", "sql")
    monkeypatch.setenv("PYICEBERG_CATALOG__SALES__URI", sales.properties["uri"])
    monkeypatch.setenv(
        "PYICEBERG_CATALOG__SALES__WAREHOUSE", sales.properties["warehouse"]
    )
    # PyIceberg reads its environment once, at import.
    monkeypatch.setattr(pyiceberg.catalog, "_ENV_CONFIG", Config())

    table = resolve_catalog("sales").load_table("ns.t")

    assert table.scan().to_arrow().num_rows == 2


def test_default_resolver_failure_is_compute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_catalog(name: str, **_: Any) -> Catalog:
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(pyiceberg.catalog, "load_catalog", load_catalog)

    with pytest.raises(
        ComputeError, match="failed to load Iceberg catalog 'sales'"
    ) as excinfo:
        resolve_catalog("sales")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_resolver_error_passes_through() -> None:
    def resolver(name: str) -> Catalog:
        msg = "not connected"
        raise ComputeError(msg)

    pl.Config.set_iceberg_catalog_resolver(resolver)

    with pytest.raises(ComputeError, match=r"^not connected$"):
        resolve_catalog("sales")


def test_resolver_returning_non_catalog_is_type_error() -> None:
    pl.Config.set_iceberg_catalog_resolver(lambda _: object())  # type: ignore[arg-type, return-value]

    with pytest.raises(TypeError, match="returned 'object' for catalog 'sales'"):
        resolve_catalog("sales")


def test_resolver_returning_noop_catalog_is_type_error() -> None:
    pl.Config.set_iceberg_catalog_resolver(lambda _: NoopCatalog("x"))

    with pytest.raises(TypeError, match="NoopCatalog for catalog 'sales'"):
        resolve_catalog("sales")


def test_check_catalog_name_rejects_empty() -> None:
    assert check_catalog_name("sales") == "sales"
    with pytest.raises(ValueError, match="must not be empty"):
        check_catalog_name("")
