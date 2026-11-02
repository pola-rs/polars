from __future__ import annotations

import io
import pickle
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

pytest.importorskip("pyiceberg.catalog.sql")

import pyiceberg.catalog
from pyiceberg.catalog.noop import NoopCatalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.utils.config import Config

from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._catalog_resolver import check_catalog_name, resolve_catalog
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    IcebergCatalogTableDescriptor,
    IcebergScanTableSerializer,
    IcebergTableWrap,
)
from polars.io.iceberg._sink import IcebergSinkState

pytestmark = pytest.mark.write_disk

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


@pytest.fixture
def _pyiceberg_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    # Results must not depend on whether `polars_iceberg` is installed.
    monkeypatch.setenv("POLARS_ICEBERG_PLANNER", "pyiceberg")


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_does_not_resolve_at_build() -> None:
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)

    assert isinstance(pl.scan_iceberg("ns.t", catalog="sales"), pl.LazyFrame)


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_resolves_at_collect(sales: SqlCatalog) -> None:
    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)

    df = pl.scan_iceberg("ns.t", catalog="sales").collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_serializes_only_the_name(sales: SqlCatalog) -> None:
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)
    data = pl.scan_iceberg("ns.t", catalog="sales").serialize()

    assert b"sales" in data
    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    df = pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_uses_resolver_active_at_collect(sales: SqlCatalog) -> None:
    with pl.Config(set_iceberg_catalog_resolver=fail_resolver):
        lf = pl.scan_iceberg("ns.t", catalog="sales")

    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)

    assert_frame_equal(lf.collect(), pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_two_named_catalogs_in_one_query(
    sales: SqlCatalog, tmp_path: Path
) -> None:
    east = new_catalog(tmp_path, "east", pl.DataFrame({"a": [3]}))
    catalogs = {"sales": sales, "east": east}
    names: list[str] = []

    def resolver(name: str) -> Catalog:
        names.append(name)
        return catalogs[name]

    pl.Config.set_iceberg_catalog_resolver(resolver)
    df = pl.concat(
        [
            pl.scan_iceberg("ns.t", catalog="sales"),
            pl.scan_iceberg("ns.t", catalog="east"),
        ]
    ).collect()

    assert_frame_equal(df.sort("a"), pl.DataFrame({"a": [1, 2, 3]}))
    assert set(names) == {"sales", "east"}


def test_scan_empty_catalog_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        pl.scan_iceberg("ns.t", catalog="")


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_static_path_ignores_catalog_name(sales: SqlCatalog) -> None:
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)
    metadata_location = sales.load_table("ns.t").metadata_location

    df = pl.scan_iceberg(metadata_location, catalog="sales").collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))


def test_descriptor_without_catalog_name_field_unpickles(sales: SqlCatalog) -> None:
    descriptor = IcebergCatalogTableDescriptor(
        table_identifier="ns.t",
        catalog_config=IcebergCatalogConfig.from_catalog(sales),
    )
    # As pickled by a polars that predates catalog names.
    del descriptor.__dict__["catalog_name"]
    descriptor = pickle.loads(pickle.dumps(descriptor))

    table = IcebergTableWrap(
        table_=NoPickleOption(None),
        table_descriptor_=descriptor,
        serializer=IcebergScanTableSerializer(),
        iceberg_storage_properties=None,
    ).get()

    assert table.name() == ("ns", "t")


def table_rows(catalog: SqlCatalog) -> pl.DataFrame:
    return pl.scan_iceberg(catalog.load_table("ns.t")).collect().sort("a")


def test_sink_state_does_not_resolve_at_build() -> None:
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)

    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")

    assert isinstance(state.attach_sink(pl.LazyFrame({"a": [3]})), pl.LazyFrame)


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_iceberg_appends_through_resolver(sales: SqlCatalog) -> None:
    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)

    pl.LazyFrame({"a": [3]}).sink_iceberg("ns.t", mode="append", catalog="sales")

    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2, 3]}))
    assert "sales" in names


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_plan_serializes_only_the_name(sales: SqlCatalog) -> None:
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    data = state.attach_sink(pl.LazyFrame({"a": [3]})).serialize()

    assert b"sales" in data
    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2, 3]}))
    assert names


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_surfaces_resolver_error_after_deserialize(sales: SqlCatalog) -> None:
    pl.Config.set_iceberg_catalog_resolver(fail_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    data = state.attach_sink(pl.LazyFrame({"a": [3]})).serialize()

    def refusing(name: str) -> Catalog:
        msg = "can only be resolved while planning"
        raise ComputeError(msg)

    pl.Config.set_iceberg_catalog_resolver(refusing)

    with pytest.raises(ComputeError, match=r"^can only be resolved while planning$"):
        pl.LazyFrame.deserialize(io.BytesIO(data)).collect()
    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2]}))


def test_sink_catalog_name_with_table_target(sales: SqlCatalog) -> None:
    with pytest.raises(TypeError, match="requires a table identifier"):
        IcebergSinkState.new(sales.load_table("ns.t"), catalog="sales")


def test_sink_empty_catalog_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IcebergSinkState.new("ns.t", catalog="")


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_state_resolves_again_after_pickling(sales: SqlCatalog) -> None:
    # What a commit task in another process receives: the state after planning.
    resolver, _ = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    state._attach_resolved_sink(pl.LazyFrame({"a": [3]})._ldf)
    data = pickle.dumps(state)

    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording(sales)
    pl.Config.set_iceberg_catalog_resolver(resolver)
    pickle.loads(data).table()
    assert names == ["sales"]

    def refusing(name: str) -> Catalog:
        msg = "can only be resolved while planning"
        raise ComputeError(msg)

    pl.Config.set_iceberg_catalog_resolver(refusing)
    with pytest.raises(ComputeError, match=r"^can only be resolved while planning$"):
        pickle.loads(data).commit([])
