from __future__ import annotations

import io
import os
import pickle
import subprocess
import sys
import warnings
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

pytest.importorskip("pyiceberg.catalog.sql")

import pyiceberg.catalog
from pyiceberg.catalog.noop import NoopCatalog
from pyiceberg.catalog.sql import SqlCatalog

from polars.io.iceberg import _catalog_resolver
from polars.io.iceberg._catalog_resolver import (
    check_catalog_name,
    resolve_catalog,
    set_iceberg_catalog_resolver,
)
from polars.io.iceberg._sink import IcebergSinkState
from tests.unit.io.conftest import (
    failing_catalog_resolver,
    recording_catalog_resolver,
)

pytestmark = [
    pytest.mark.write_disk,
    pytest.mark.usefixtures("reset_iceberg_catalog_resolver"),
]

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pyiceberg.catalog import Catalog


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


def test_set_resolver_rejects_non_callable() -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)

    with pytest.raises(TypeError, match="must be callable or None; found 'dict'"):
        set_iceberg_catalog_resolver({"sales": None})  # type: ignore[arg-type]
    assert _catalog_resolver.ICEBERG_CATALOG_RESOLVER is failing_catalog_resolver


def test_resolve_catalog_uses_installed_resolver(sales: SqlCatalog) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

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
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    set_iceberg_catalog_resolver(None)

    assert resolve_catalog("sales") is sales
    assert loaded == ["sales"]


@pytest.mark.slow
def test_default_resolver_reads_pyiceberg_env(sales: SqlCatalog) -> None:
    # PyIceberg reads its environment when it is imported, so in a new process.
    env = {
        **os.environ,
        "PYICEBERG_CATALOG__SALES__TYPE": "sql",
        "PYICEBERG_CATALOG__SALES__URI": sales.properties["uri"],
        "PYICEBERG_CATALOG__SALES__WAREHOUSE": sales.properties["warehouse"],
    }
    script = (
        "from polars.io.iceberg._catalog_resolver import resolve_catalog\n"
        "table = resolve_catalog('sales').load_table('ns.t')\n"
        "print(table.scan().to_arrow().num_rows)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"


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

    set_iceberg_catalog_resolver(resolver)

    # pytest matches against the message and its notes, so check each apart.
    with pytest.raises(ComputeError, match="not connected") as excinfo:
        resolve_catalog("sales")
    assert str(excinfo.value) == "not connected"
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sales'"
    ]


def test_resolver_error_names_the_catalog() -> None:
    # A mapping's `__getitem__` raises a KeyError naming only the key.
    catalogs: dict[str, Catalog] = {}
    set_iceberg_catalog_resolver(catalogs.__getitem__)

    with pytest.raises(KeyError) as excinfo:
        resolve_catalog("sale")
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sale'"
    ]


def test_resolver_returning_non_catalog_is_type_error() -> None:
    set_iceberg_catalog_resolver(lambda _: object())  # type: ignore[arg-type, return-value]

    with pytest.raises(TypeError, match="returned 'object' for catalog 'sales'"):
        resolve_catalog("sales")


def test_resolver_returning_noop_catalog_is_type_error() -> None:
    set_iceberg_catalog_resolver(lambda _: NoopCatalog("x"))

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
    set_iceberg_catalog_resolver(failing_catalog_resolver)

    assert isinstance(pl.scan_iceberg("ns.t", catalog="sales"), pl.LazyFrame)


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_resolves_at_collect(sales: SqlCatalog) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

    df = pl.scan_iceberg("ns.t", catalog="sales").collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_serializes_only_the_name(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    data = pl.scan_iceberg("ns.t", catalog="sales").serialize()

    assert b"sales" in data
    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    df = pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_uses_resolver_active_at_collect(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    lf = pl.scan_iceberg("ns.t", catalog="sales")

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

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

    set_iceberg_catalog_resolver(resolver)
    df = pl.concat(
        [
            pl.scan_iceberg("ns.t", catalog="sales"),
            pl.scan_iceberg("ns.t", catalog="east"),
        ]
    ).collect()

    assert_frame_equal(df.sort("a"), pl.DataFrame({"a": [1, 2, 3]}))
    assert set(names) == {"sales", "east"}


@pytest.mark.usefixtures("_pyiceberg_planner")
@pytest.mark.parametrize(
    "load",
    [pl.LazyFrame.collect_schema, pl.LazyFrame.explain, pl.LazyFrame.collect],
    ids=["collect_schema", "explain", "collect"],
)
def test_scan_serializes_only_the_name_after_loading_the_table(
    sales: SqlCatalog, load: Callable[[pl.LazyFrame], Any]
) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    lf = pl.scan_iceberg("ns.t", catalog="sales")
    load(lf)
    assert names == ["sales"]

    # The table loaded here is not pinned: no metadata path replaces the name.
    data = lf.serialize()
    assert b"metadata.json" not in data
    assert b"sqlite" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    df = pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


def test_scan_empty_catalog_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        pl.scan_iceberg("ns.t", catalog="")


def test_scan_catalog_name_with_static_path(sales: SqlCatalog) -> None:
    metadata_location = sales.load_table("ns.t").metadata_location

    with pytest.raises(TypeError, match="requires a table identifier"):
        pl.scan_iceberg(metadata_location, catalog="sales")


def test_scan_catalog_name_with_table_source(sales: SqlCatalog) -> None:
    with pytest.raises(TypeError, match="requires a table identifier"):
        pl.scan_iceberg(sales.load_table("ns.t"), catalog="sales")


def table_rows(catalog: SqlCatalog) -> pl.DataFrame:
    return pl.scan_iceberg(catalog.load_table("ns.t")).collect().sort("a")


def test_sink_state_does_not_resolve_at_build() -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)

    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")

    assert isinstance(state.attach_sink(pl.LazyFrame({"a": [3]})), pl.LazyFrame)


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_iceberg_appends_through_resolver(sales: SqlCatalog) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

    pl.LazyFrame({"a": [3]}).sink_iceberg("ns.t", mode="append", catalog="sales")

    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2, 3]}))
    assert "sales" in names


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_plan_serializes_only_the_name(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    data = state.attach_sink(pl.LazyFrame({"a": [3]})).serialize()

    assert b"sales" in data
    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2, 3]}))
    assert names


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_surfaces_resolver_error_after_deserialize(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    data = state.attach_sink(pl.LazyFrame({"a": [3]})).serialize()

    def refusing(name: str) -> Catalog:
        msg = "can only be resolved while planning"
        raise ComputeError(msg)

    set_iceberg_catalog_resolver(refusing)

    with pytest.raises(
        ComputeError, match="can only be resolved while planning"
    ) as excinfo:
        pl.LazyFrame.deserialize(io.BytesIO(data)).collect()
    assert str(excinfo.value) == "can only be resolved while planning"
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sales'"
    ]
    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2]}))


def test_sink_state_for_catalog_name_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `sink_iceberg` is unstable as a whole and warns so itself.
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IcebergSinkState.new("ns.t", mode="append", catalog="sales")


@pytest.mark.parametrize(
    "field", ["py_catalog_class_module", "py_catalog_class_qualname"]
)
def test_sink_state_rejects_half_set_catalog_class(field: str) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    setattr(state, field, "SqlCatalog")

    with pytest.raises(ValueError, match="must both be set, or both be None"):
        state.table()


def test_sink_catalog_name_with_table_target(sales: SqlCatalog) -> None:
    with pytest.raises(TypeError, match="requires a table identifier"):
        IcebergSinkState.new(sales.load_table("ns.t"), catalog="sales")


def test_sink_empty_catalog_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IcebergSinkState.new("ns.t", catalog="")


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_state_resolves_again_after_pickling(sales: SqlCatalog) -> None:
    # What a commit task in another process receives: the state after planning.
    resolver, _ = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    state._attach_resolved_sink(pl.LazyFrame({"a": [3]})._ldf)
    data = pickle.dumps(state)

    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    pickle.loads(data).table()
    assert names == ["sales"]

    def refusing(name: str) -> Catalog:
        msg = "can only be resolved while planning"
        raise ComputeError(msg)

    set_iceberg_catalog_resolver(refusing)
    with pytest.raises(
        ComputeError, match="can only be resolved while planning"
    ) as excinfo:
        pickle.loads(data).commit([])
    assert str(excinfo.value) == "can only be resolved while planning"
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sales'"
    ]
