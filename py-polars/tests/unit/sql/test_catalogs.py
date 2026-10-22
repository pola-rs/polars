from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal

pytest.importorskip("pyiceberg.catalog.sql")

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError


class CountingCatalog(SqlCatalog):  # type: ignore[misc]
    """A SQL catalog that records the tables it loads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loads: list[Any] = []

    def load_table(self, identifier: Any) -> Any:
        self.loads.append(identifier)
        return super().load_table(identifier)


TABLES = {
    ("tpch", "lineitem"): pl.DataFrame({"k": [1, 2], "v": ["a", "b"]}),
    ("tpch", "orders"): pl.DataFrame({"k": [1, 2, 3], "o": [10, 20, 30]}),
    ("CH0010", "lineitem"): pl.DataFrame({"k": [7], "v": ["upper"]}),
    ("prod", "tpch", "lineitem"): pl.DataFrame({"k": [9], "v": ["nested"]}),
}


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


@pytest.fixture
def lake(tmp_path: Path) -> CountingCatalog:
    catalog = CountingCatalog(
        "lake",
        uri=f"sqlite:///{tmp_path / 'catalog.db'}",
        warehouse=f"file://{tmp_path}",
    )
    for identifier, df in TABLES.items():
        catalog.create_namespace_if_not_exists(identifier[:-1])
        table = catalog.create_table(identifier, schema=df.to_arrow().schema)
        df.write_iceberg(table, mode="append")
    catalog.loads.clear()
    return catalog


def lake_ctx(catalog: CountingCatalog, **kwargs: Any) -> pl.SQLContext[pl.LazyFrame]:
    kwargs.setdefault("default_namespace", "tpch")
    kwargs.setdefault("default", True)
    return pl.SQLContext().register_catalog("lake", catalog, **kwargs)


LINEITEM = pl.DataFrame({"k": [1, 2], "v": ["a", "b"]})


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM lineitem",
        "SELECT * FROM tpch.lineitem",
        "SELECT * FROM lake.tpch.lineitem",
        "SELECT * FROM LAKE.tpch.lineitem",
        "SELECT lineitem.k, lineitem.v FROM lineitem",
        "SELECT tpch.lineitem.k, tpch.lineitem.v FROM lineitem",
        "SELECT lake.tpch.lineitem.k, lake.tpch.lineitem.v FROM lineitem",
        "SELECT li.k, li.v FROM tpch.lineitem AS li",
        "WITH c AS (SELECT * FROM lineitem) SELECT * FROM c",
        "SELECT * FROM (SELECT * FROM lineitem) AS sub",
    ],
)
def test_catalog_table_names(lake: CountingCatalog, query: str) -> None:
    res = lake_ctx(lake).execute(query, eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)


@pytest.mark.write_disk
def test_catalog_table_deferred_plan_round_trip(lake: CountingCatalog) -> None:
    # the full name survives serialization, so the long qualifier still resolves
    lf = lake_ctx(lake).execute(
        "SELECT lake.tpch.lineitem.v FROM lineitem WHERE lineitem.k > 1"
    )
    assert lake.loads == []
    for res in (
        lf,
        pickle.loads(pickle.dumps(lf)),
        pl.LazyFrame.deserialize(io.BytesIO(lf.serialize())),
    ):
        assert res.collect().rows() == [("b",)]


@pytest.mark.write_disk
def test_catalog_namespaces(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    assert ctx.execute("SELECT v FROM CH0010.lineitem", eager=True).item() == "upper"
    assert (
        ctx.execute("SELECT v FROM prod.tpch.lineitem", eager=True).item() == "nested"
    )

    for namespace in ("prod.tpch", ["prod", "tpch"]):
        ctx = lake_ctx(lake, default_namespace=namespace)
        assert ctx.execute("SELECT v FROM lineitem", eager=True).item() == "nested"

    with pytest.raises(ValueError, match="invalid `default_namespace`"):
        lake_ctx(lake, default_namespace="")


@pytest.mark.write_disk
def test_catalog_identifiers_are_case_sensitive(lake: CountingCatalog) -> None:
    lf = lake_ctx(lake).execute("SELECT * FROM ch0010.lineitem")
    with pytest.raises(
        Exception,
        match=r"relation 'lake\.ch0010\.lineitem' not found: .*case-sensitive",
    ):
        lf.collect()


@pytest.mark.write_disk
def test_catalog_without_default(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake, default=False)
    assert (
        ctx.execute("SELECT COUNT(*) FROM lake.tpch.lineitem", eager=True).item() == 2
    )
    for name in ("lineitem", "tpch.lineitem"):
        with pytest.raises(SQLInterfaceError, match=f"relation '{name}' was not found"):
            ctx.execute(f"SELECT * FROM {name}", eager=True)

    # the hint is given where the attached catalogs are known (here: the eager path)
    with pytest.raises(SQLInterfaceError, match="no default catalog is set"):
        ctx.execute("CREATE TABLE t AS SELECT * FROM lineitem")


@pytest.mark.write_disk
def test_catalog_without_default_namespace(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake, default_namespace=None)
    assert ctx.execute("SELECT COUNT(*) FROM tpch.lineitem", eager=True).item() == 2
    with pytest.raises(SQLInterfaceError, match="relation 'lineitem' was not found"):
        ctx.execute("SELECT * FROM lineitem", eager=True)
    with pytest.raises(
        SQLInterfaceError, match="catalog 'lake' has no default namespace"
    ):
        ctx.execute("CREATE TABLE t AS SELECT * FROM lineitem")


@pytest.mark.write_disk
def test_register_catalog_names(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    # the same name replaces the catalog
    ctx.register_catalog("lake", lake, default_namespace="CH0010", default=True)
    assert ctx.execute("SELECT v FROM lineitem", eager=True).item() == "upper"
    # ... with the new settings, and keeps its default status
    ctx.register_catalog("lake", lake, default_namespace="tpch", default=False)
    assert ctx.execute("SELECT COUNT(*) FROM lineitem", eager=True).item() == 2

    with pytest.raises(pl.exceptions.InvalidOperationError, match="only in case"):
        ctx.register_catalog("LAKE", lake)

    for obj in (object(), pl.LazyFrame()):
        with pytest.raises(TypeError, match="PyIceberg catalog"):
            ctx.register_catalog("other", obj)  # type: ignore[arg-type]

    # a later default replaces the earlier one; re-registering a catalog that is not
    # the default leaves the default alone; unregistering the default leaves none
    ctx.register_catalog("other", lake, default_namespace="tpch", default=True)
    ctx.register_catalog("lake", lake, default_namespace="CH0010")
    assert ctx.execute("SELECT COUNT(*) FROM lineitem", eager=True).item() == 2
    ctx.unregister_catalog("OTHER")
    with pytest.raises(SQLInterfaceError, match="relation 'lineitem' was not found"):
        ctx.execute("SELECT * FROM lineitem", eager=True)
    assert (
        ctx.execute("SELECT v FROM lake.CH0010.lineitem", eager=True).item() == "upper"
    )

    ctx.unregister_catalog("lake")
    with pytest.raises(SQLInterfaceError, match="was not found"):
        ctx.execute("SELECT * FROM lake.tpch.lineitem", eager=True)


@pytest.mark.write_disk
def test_local_names_win_over_catalog(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.register("lineitem", pl.LazyFrame({"k": [0]}))
    assert ctx.execute("SELECT k FROM lineitem", eager=True).item() == 0
    ctx.unregister("lineitem")

    res = ctx.execute(
        "WITH lineitem AS (SELECT 5 AS k) SELECT k FROM lineitem", eager=True
    )
    assert res.item() == 5
    # a CTE in a subquery shadows only there
    res = ctx.execute(
        """
        SELECT COUNT(*) FROM lineitem
        WHERE k IN (WITH lineitem AS (SELECT 2 AS k) SELECT k FROM lineitem)
        """,
        eager=True,
    )
    assert res.item() == 1
    assert ctx.tables() == []


@pytest.mark.write_disk
def test_catalog_tables_load_lazily(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    lf = ctx.execute(
        "SELECT * FROM lineitem JOIN orders USING (k) JOIN lineitem AS l2 USING (k)"
    )
    assert lake.loads == []
    assert lf.collect().height == 2
    # one load per table, also for a self-join
    assert sorted(lake.loads) == [("tpch", "lineitem"), ("tpch", "orders")]

    # also through a derived table
    lake.loads.clear()
    lf = ctx.execute("SELECT * FROM (SELECT * FROM orders) AS x")
    assert lake.loads == []
    assert lf.collect().height == 3
    assert lake.loads == [("tpch", "orders")]


@pytest.mark.write_disk
def test_catalog_table_spelled_twice(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    from_clause = "FROM lineitem CROSS JOIN tpch.lineitem"
    assert ctx.execute(f"SELECT * {from_clause}", eager=True).height == 4
    with pytest.raises(SQLInterfaceError, match="appears more than once"):
        ctx.execute(f"SELECT lineitem.k {from_clause}", eager=True)


@pytest.mark.write_disk
def test_catalog_tables_in_subqueries(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    res = ctx.execute(
        """
        SELECT o FROM orders
        WHERE EXISTS (SELECT 1 FROM lineitem WHERE lineitem.k = orders.k)
        ORDER BY o
        """,
        eager=True,
    )
    assert res.to_series().to_list() == [10, 20]

    res = ctx.execute(
        "SELECT o FROM orders WHERE k > (SELECT MIN(k) FROM tpch.lineitem) ORDER BY o",
        eager=True,
    )
    assert res.to_series().to_list() == [20, 30]


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE lake.tpch.t AS SELECT 1 AS x",
        "CREATE TABLE tpch.t AS SELECT 1 AS x",
        "CREATE TABLE t AS SELECT 1 AS x",
        "CREATE TABLE IF NOT EXISTS tpch.t (x INT)",
        "TRUNCATE TABLE lake.tpch.lineitem",
        "TRUNCATE TABLE tpch.lineitem",
        "TRUNCATE TABLE IF EXISTS lake.tpch.lineitem",
        "DELETE FROM lineitem WHERE k = 1",
        "DELETE FROM lake.tpch.lineitem",
    ],
)
def test_catalog_ddl_is_not_supported(lake: CountingCatalog, statement: str) -> None:
    with pytest.raises(
        SQLInterfaceError, match="attached catalog 'lake' is not supported"
    ):
        lake_ctx(lake).execute(statement)


@pytest.mark.write_disk
def test_create_table_resolves_like_a_read(lake: CountingCatalog) -> None:
    # #29892: a name of the default catalog doesn't silently create a registered table
    ctx = lake_ctx(lake)
    ctx.register("t", pl.LazyFrame({"x": [1]}))
    for statement, name in (
        ("CREATE TABLE sales.t2 (id BIGINT)", "'sales.t2' is 'lake.sales.t2'"),
        ("CREATE TABLE t AS SELECT * FROM lineitem", "'t' is 'lake.tpch.t'"),
    ):
        with pytest.raises(SQLInterfaceError, match=name):
            ctx.execute(statement)
    assert ctx.tables() == ["t"]

    # a name that no catalog applies to creates a registered table
    ctx = lake_ctx(lake, default_namespace=None)
    ctx.execute("CREATE TABLE t AS SELECT 1 AS x")
    with pytest.raises(SQLInterfaceError, match="'lake' has no default namespace"):
        ctx.execute("CREATE TABLE lake.t (x INT)")
    ctx = lake_ctx(lake, default=False)
    ctx.execute("CREATE TABLE tpch.t AS SELECT 1 AS x")
    assert ctx.execute("SELECT x FROM tpch.t", eager=True).item() == 1


@pytest.mark.write_disk
def test_if_exists_without_catalog_name_acts_on_registered_tables(
    lake: CountingCatalog,
) -> None:
    # a cleanup script keeps working once a default catalog is attached
    ctx = lake_ctx(lake)
    ctx.register("tmp", pl.LazyFrame({"x": [1]}))
    ctx.register("staged", pl.LazyFrame({"x": [1]}))
    for statement in (
        "DROP TABLE IF EXISTS tmp",
        "DROP TABLE IF EXISTS tmp",
        "TRUNCATE TABLE IF EXISTS staged",
        "TRUNCATE TABLE IF EXISTS tpch.lineitem",
    ):
        ctx.execute(statement)
    assert ctx.tables() == ["staged"]
    assert ctx.execute("SELECT * FROM staged", eager=True).is_empty()
    assert lake.loads == []
    assert ctx.execute("SELECT COUNT(*) FROM lineitem", eager=True).item() == 2

    # a name that fails leaves the other names of the statement in place
    ctx.register_many(Foo=pl.LazyFrame({"x": [1]}), FOO=pl.LazyFrame({"x": [1]}))
    with pytest.raises(SQLInterfaceError, match="more than one table ignoring case"):
        ctx.execute("DROP TABLE IF EXISTS staged, foo")
    assert sorted(ctx.tables()) == ["FOO", "Foo", "staged"]


@pytest.mark.write_disk
def test_catalogs_attached_in_scope_are_detached(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake, default=False)

    def count(name: str) -> int:
        return int(ctx.execute(f"SELECT COUNT(*) FROM {name}", eager=True).item())

    with ctx:
        ctx.register_catalog("other", lake)
        with ctx:
            ctx.register_catalog("inner", lake)
            assert count("inner.tpch.lineitem") == 2
        assert count("other.tpch.lineitem") == 2
        with pytest.raises(SQLInterfaceError, match=r"'inner\.tpch\.lineitem' was not"):
            count("inner.tpch.lineitem")
    with pytest.raises(SQLInterfaceError, match=r"'other\.tpch\.lineitem' was not"):
        count("other.tpch.lineitem")
    # attached before the scope: kept
    assert count("lake.tpch.lineitem") == 2


@pytest.mark.write_disk
def test_scope_exit_restores_default_catalog(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)

    def lineitem_rows() -> int:
        return int(ctx.execute("SELECT COUNT(*) FROM lineitem", eager=True).item())

    with ctx:
        ctx.register_catalog("tmp", lake, default_namespace="CH0010", default=True)
        with ctx:
            ctx.register_catalog("inner", lake, default_namespace="tpch", default=True)
        assert lineitem_rows() == 1
    assert lineitem_rows() == 2

    # a default moved to a catalog attached before the scope stays
    ctx.register_catalog("other", lake, default_namespace="CH0010")
    with ctx:
        ctx.register_catalog("other", lake, default_namespace="CH0010", default=True)
    assert lineitem_rows() == 1


@pytest.mark.write_disk
def test_ddl_on_local_table_named_like_catalog(lake: CountingCatalog) -> None:
    # a registered table with the whole dotted name is dropped instead
    ctx = lake_ctx(lake)
    ctx.register("lake.tpch.mine", pl.LazyFrame({"x": [1]}))
    ctx.register("lineitem", pl.LazyFrame({"x": [1]}))
    ctx.execute("DROP TABLE lake.tpch.mine")
    ctx.execute("DROP TABLE lineitem")
    assert ctx.tables() == []
    assert lake.table_exists(("tpch", "lineitem"))


def data_files(catalog: CountingCatalog, *identifier: str) -> list[Path]:
    location = catalog.load_table(identifier).location().removeprefix("file://")
    return list(Path(location).rglob("*.parquet"))


@pytest.mark.write_disk
@pytest.mark.parametrize("name", ["lake.tpch.lineitem", "tpch.lineitem", "lineitem"])
def test_drop_catalog_table(lake: CountingCatalog, name: str) -> None:
    # #29896
    ctx = lake_ctx(lake)
    files = data_files(lake, "tpch", "lineitem")
    assert files
    res = ctx.execute(f"DROP TABLE {name}", eager=True)
    assert res.item() == "DROP TABLE lake.tpch.lineitem"
    assert not lake.table_exists(("tpch", "lineitem"))
    assert lake.table_exists(("tpch", "orders"))
    assert all(file.exists() for file in files)

    ctx.execute(f"DROP TABLE IF EXISTS {name}")
    with pytest.raises(NoSuchTableError, match=f"table '{name}' does not exist"):
        ctx.execute(f"DROP TABLE {name}")


@pytest.mark.write_disk
def test_drop_catalog_table_purge(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    files = data_files(lake, "tpch", "orders")
    assert files
    res = ctx.execute("DROP TABLE orders PURGE", eager=True)
    assert res.item() == "DROP TABLE lake.tpch.orders PURGE"
    assert not lake.table_exists(("tpch", "orders"))
    assert not any(file.exists() for file in files)
    res = ctx.execute("DROP TABLE IF EXISTS orders PURGE", eager=True)
    assert res.item() == "DROP TABLE lake.tpch.orders PURGE"


@pytest.mark.write_disk
def test_drop_catalog_table_from_catalog_config(lake: CountingCatalog) -> None:
    from polars.io.iceberg import IcebergCatalogConfig

    config = IcebergCatalogConfig.from_catalog(lake)
    ctx = pl.SQLContext().register_catalog(
        "lake", config, default_namespace="tpch", default=True
    )
    ctx.execute("DROP TABLE orders")
    assert not lake.table_exists(("tpch", "orders"))


@pytest.mark.write_disk
def test_drop_registered_table_differing_in_case(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.register("LINEITEM", pl.LazyFrame({"x": [1]}))
    ctx.execute("DROP TABLE IF EXISTS lineitem")
    assert ctx.tables() == []
    assert lake.table_exists(("tpch", "lineitem"))


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("statement", "error"),
    [
        ("DROP TABLE lineitem, orders", "takes one table name"),
        ("DROP TABLE tmp, lineitem", "takes one table name"),
        ("DROP TABLE tmp PURGE", "PURGE` only applies to catalog tables"),
        ("DROP TABLE lineitem CASCADE", "CASCADE` is not supported"),
    ],
)
def test_drop_catalog_table_errors(
    lake: CountingCatalog, statement: str, error: str
) -> None:
    ctx = lake_ctx(lake)
    ctx.register("tmp", pl.LazyFrame({"x": [1]}))
    with pytest.raises(SQLInterfaceError, match=error):
        ctx.execute(statement)
    assert ctx.tables() == ["tmp"]
    assert lake.table_exists(("tpch", "lineitem"))
    assert lake.table_exists(("tpch", "orders"))


@pytest.mark.write_disk
def test_drop_errors_in_name_order(lake: CountingCatalog) -> None:
    # names that no catalog resolves report their errors in statement order
    ctx = lake_ctx(lake, default=False)
    ctx.register_many({"Foo.x": pl.LazyFrame(), "FOO.X": pl.LazyFrame()})
    with pytest.raises(SQLInterfaceError, match="table 'missing' does not exist"):
        ctx.execute("DROP TABLE missing, foo.x")


@pytest.mark.write_disk
def test_drop_missing_catalog_table(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    for name in ("nope", "nope.t", "LINEITEM"):
        ctx.execute(f"DROP TABLE IF EXISTS {name}")
    with pytest.raises(NoSuchTableError, match="identifiers are case-sensitive"):
        ctx.execute("DROP TABLE LINEITEM")
    assert lake.table_exists(("tpch", "lineitem"))


@pytest.mark.write_disk
def test_catalog_name_of_registered_table(lake: CountingCatalog) -> None:
    # `lineitem` refers to catalog table `lake.tpch.lineitem`, which a registered
    # table also names; reading either for the other would be silently wrong
    ctx = lake_ctx(lake)
    ctx.register("lake.tpch.lineitem", pl.LazyFrame({"k": [0], "v": ["local"]}))
    assert ctx.execute("SELECT v FROM lake.tpch.lineitem", eager=True).item() == "local"
    for query in (
        "SELECT * FROM lineitem",
        "SELECT * FROM lake.tpch.lineitem UNION ALL SELECT * FROM lineitem",
    ):
        with pytest.raises(
            SQLInterfaceError, match="also the name of a registered table"
        ):
            ctx.execute(query, eager=True)


@pytest.mark.write_disk
def test_catalog_names_differing_in_dot_placement(lake: CountingCatalog) -> None:
    for identifier, k in ((("tpch.foo", "t"), 11), (("tpch", "foo.t"), 22)):
        lake.create_namespace_if_not_exists(identifier[:-1])
        df = pl.DataFrame({"k": [k]})
        table = lake.create_table(identifier, schema=df.to_arrow().schema)
        df.write_iceberg(table, mode="append")

    ctx = lake_ctx(lake)
    assert ctx.execute('SELECT k FROM lake."tpch.foo".t', eager=True).item() == 11
    assert ctx.execute('SELECT k FROM lake.tpch."foo.t"', eager=True).item() == 22
    with pytest.raises(SQLInterfaceError, match="same dotted name"):
        ctx.execute(
            'SELECT a.k, b.k FROM lake."tpch.foo".t a CROSS JOIN lake.tpch."foo.t" b',
            eager=True,
        )


@pytest.mark.write_disk
def test_cte_visible_after_its_definition(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    # inside its own definition, `lineitem` is still the catalog table
    res = ctx.execute(
        "WITH lineitem AS (SELECT * FROM lineitem) SELECT * FROM lineitem", eager=True
    )
    assert_frame_equal(res, LINEITEM, check_row_order=False)
    # a later CTE does not shadow a table read by an earlier one
    res = ctx.execute(
        "WITH a AS (SELECT * FROM orders), orders AS (SELECT 1 AS k) "
        "SELECT (SELECT COUNT(*) FROM a) AS n_a, (SELECT COUNT(*) FROM orders) AS n_o",
        eager=True,
    )
    assert res.row(0) == (3, 1)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "query",
    [
        "SELECT lake.lineitem.k, lake.lineitem.v FROM lake.lineitem",
        "SELECT lake.lineitem.* FROM lake.lineitem",
        "SELECT lake.tpch.lineitem.k, lineitem.v FROM lake.lineitem",
    ],
)
def test_qualifier_as_written(lake: CountingCatalog, query: str) -> None:
    # `lake.lineitem` (catalog and table, default namespace) qualifies as written
    ctx = lake_ctx(lake)
    lf = ctx.execute(query)
    for res in (lf, pickle.loads(pickle.dumps(lf))):
        assert_frame_equal(res.collect(), LINEITEM, check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM lineitem WHERE k < 0 UNION ALL TABLE lineitem",
        "SELECT * FROM lineitem WHERE k < 0 UNION ALL TABLE tpch.lineitem",
    ],
)
def test_table_statement_of_catalog_table(lake: CountingCatalog, query: str) -> None:
    lf = lake_ctx(lake).execute(query)
    for res in (lf, pickle.loads(pickle.dumps(lf))):
        assert_frame_equal(res.collect(), LINEITEM, check_row_order=False)


def add_table(catalog: CountingCatalog, identifier: tuple[str, ...], k: int) -> None:
    catalog.create_namespace_if_not_exists(identifier[:-1])
    df = pl.DataFrame({"k": [k]})
    table = catalog.create_table(identifier, schema=df.to_arrow().schema)
    df.write_iceberg(table, mode="append")


@pytest.mark.write_disk
def test_qualifier_as_written_prefers_exact_case(lake: CountingCatalog) -> None:
    add_table(lake, ("tpch", "LineItem"), 5)
    res = lake_ctx(lake).execute(
        "SELECT lake.lineitem.k AS lo, lake.LineItem.k AS hi "
        "FROM lake.lineitem CROSS JOIN lake.LineItem ORDER BY lo",
        eager=True,
    )
    assert res.rows() == [(1, 5), (2, 5)]


@pytest.mark.write_disk
def test_qualifier_as_written_stays_in_scope(lake: CountingCatalog) -> None:
    # `lake.lineitem` names `lake.tpch.lineitem` in the first branch only
    add_table(lake, ("lake", "lineitem"), 7)
    ctx = lake_ctx(lake).register_catalog("archive", lake)
    res = ctx.execute(
        "SELECT lake.lineitem.k FROM lake.lineitem "
        "UNION ALL SELECT lake.lineitem.k FROM archive.lake.lineitem",
        eager=True,
    )
    assert sorted(res.to_series().to_list()) == [1, 2, 7]


@pytest.mark.write_disk
def test_catalog_name_of_from_alias(lake: CountingCatalog) -> None:
    # the derived table's alias would shadow the catalog table that `lineitem` refers to
    with pytest.raises(SQLInterfaceError, match="registered table, CTE or alias"):
        lake_ctx(lake).execute(
            'SELECT b.k FROM (SELECT 42 AS k) AS "lake.tpch.lineitem" '
            "CROSS JOIN lineitem AS b",
            eager=True,
        )


@pytest.mark.write_disk
def test_catalog_name_of_derived_alias_in_other_operand(lake: CountingCatalog) -> None:
    # the derived table is registered under its alias while the statement runs
    with pytest.raises(SQLInterfaceError, match="registered table, CTE or alias"):
        lake_ctx(lake).execute(
            'SELECT k FROM (SELECT 42 AS k) AS "lake.tpch.lineitem" '
            "UNION ALL SELECT k FROM lineitem",
            eager=True,
        )


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "query",
    [
        'SELECT k FROM orders AS "lake.tpch.lineitem" UNION ALL SELECT k FROM lineitem',
        'SELECT b.k FROM orders AS "lake.tpch.lineitem" CROSS JOIN lineitem AS b',
    ],
)
def test_catalog_name_of_table_alias(lake: CountingCatalog, query: str) -> None:
    # a table alias is not looked up as a table, so `lineitem` still reads the catalog's
    res = lake_ctx(lake).execute(query, eager=True)
    expected = [1, 2, 3, 1, 2] if "UNION" in query else [1, 1, 1, 2, 2, 2]
    assert sorted(res.to_series().to_list()) == sorted(expected)


@pytest.mark.write_disk
def test_qualifier_as_written_per_occurrence(lake: CountingCatalog) -> None:
    # `lake.lineitem` written in the CTE does not name the outer `lake.tpch.lineitem`
    add_table(lake, ("lake", "lineitem"), 7)
    ctx = lake_ctx(lake).register_catalog("archive", lake)
    query = "SELECT lake.lineitem.k FROM lake.tpch.lineitem CROSS JOIN archive.lake.lineitem"
    for prefix in ("", "WITH unused AS (SELECT * FROM lake.lineitem) "):
        lf = ctx.execute(prefix + query)
        for res in (lf, pickle.loads(pickle.dumps(lf))):
            assert res.collect().to_series().to_list() == [7, 7]


@pytest.mark.write_disk
def test_catalog_name_case_variant_in_deferred_plan(lake: CountingCatalog) -> None:
    # the deferred plan does not capture `LAKE.tpch.lineitem` as a registered table
    lf = lake_ctx(lake).execute(
        "SELECT * FROM LAKE.tpch.lineitem UNION ALL SELECT * FROM lineitem"
    )
    for res in (lf, pickle.loads(pickle.dumps(lf))):
        assert res.collect().height == 4


@pytest.mark.write_disk
def test_alias_in_nested_query_does_not_collide(lake: CountingCatalog) -> None:
    # the CTE's alias lives in the CTE's scope, not where `lineitem` is read
    res = lake_ctx(lake).execute(
        'WITH x AS (SELECT * FROM orders AS "lake.tpch.lineitem") '
        "SELECT * FROM lineitem",
        eager=True,
    )
    assert_frame_equal(res, LINEITEM, check_row_order=False)
