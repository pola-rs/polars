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
@pytest.mark.parametrize("qualifier", ["lake.tpch.orders", "tpch.orders", "orders"])
def test_catalog_table_read_by_nested_subquery(
    lake: CountingCatalog, qualifier: str
) -> None:
    # The innermost subquery reads the catalog table two levels up.
    ctx = lake_ctx(lake)
    ctx.register("b", pl.DataFrame({"k": [1, 2]}))
    ctx.register("c", pl.DataFrame({"k": [1, 3]}))
    query = (
        f"SELECT {qualifier}.k FROM orders WHERE {qualifier}.k IN "
        f"(SELECT b.k FROM b WHERE b.k IN (SELECT c.k FROM c WHERE c.k = {qualifier}.k))"
    )
    assert ctx.execute(query, eager=True).rows() == [(1,)]
    assert ctx.execute(query).collect().rows() == [(1,)]
    assert ctx.execute(f"EXPLAIN {query}", eager=True).height > 0


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


def table_fields(catalog: CountingCatalog, *identifier: str) -> list[Any]:
    schema = catalog.load_table(identifier).schema()
    return [(f.name, str(f.field_type), f.required) for f in schema.fields]


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "name", ["lake.tpch.t", "tpch.t", "t", "LAKE.tpch.t", "lake.t"]
)
def test_create_catalog_table(lake: CountingCatalog, name: str) -> None:
    # #29892
    ctx = lake_ctx(lake)
    res = ctx.execute(
        f"CREATE TABLE {name} (id BIGINT NOT NULL, v VARCHAR NULL, d DATE, "
        "s STRUCT<a INT, b VARCHAR>, l INT[])",
        eager=True,
    )
    assert res.item() == "CREATE TABLE lake.tpch.t"
    assert ctx.tables() == []
    fields = table_fields(lake, "tpch", "t")
    assert [f[0] for f in fields] == ["id", "v", "d", "s", "l"]
    assert [f[2] for f in fields] == [True, False, False, False, False]
    assert fields[:3] == [
        ("id", "long", True),
        ("v", "string", False),
        ("d", "date", False),
    ]

    # the new table is read like any catalog table, and can be written to
    schema = pl.Schema(
        {
            "id": pl.Int64(),
            "v": pl.String(),
            "d": pl.Date(),
            "s": pl.Struct({"a": pl.Int32, "b": pl.String}),
            "l": pl.List(pl.Int32),
        }
    )
    assert ctx.execute("SELECT * FROM t", eager=True).schema == schema
    df = pl.DataFrame(
        {"id": [1], "v": ["x"], "d": [None], "s": [None], "l": [[1, 2]]},
        schema=schema,
    )
    df.lazy().sink_iceberg(lake.load_table(("tpch", "t")), mode="append")
    assert_frame_equal(ctx.execute("SELECT * FROM t", eager=True), df)


@pytest.mark.write_disk
def test_create_catalog_table_if_not_exists(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import TableAlreadyExistsError

    ctx = lake_ctx(lake)
    res = ctx.execute("CREATE TABLE IF NOT EXISTS lineitem (x INT)", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.lineitem"
    res = ctx.execute("SELECT * FROM lineitem", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)
    with pytest.raises(
        TableAlreadyExistsError,
        match=r"table 'lineitem' already exists: catalog table 'tpch\.lineitem'",
    ):
        ctx.execute("CREATE TABLE lineitem (x INT)")

    # the columns are checked also when the table exists
    with pytest.raises(TypeError, match="columns without a lossless Iceberg type"):
        ctx.execute("CREATE TABLE IF NOT EXISTS lineitem (x UBIGINT)")

    # a table created after the existence check
    monkeypatch.setattr(lake, "table_exists", lambda identifier: False)
    res = ctx.execute("CREATE TABLE IF NOT EXISTS lineitem (x INT)", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.lineitem"
    assert table_fields(lake, "tpch", "lineitem")[0][0] == "k"


@pytest.mark.write_disk
def test_create_catalog_table_options(lake: CountingCatalog, tmp_path: Path) -> None:
    ctx = lake_ctx(lake)
    location = f"file://{tmp_path / 'custom'}"
    ctx.execute(
        f"CREATE TABLE t1 (x INT) WITH ('location' = '{location}', "
        "'format-version' = 1, 'write.parquet.compression-codec' = 'zstd', "
        "flag = true, n = -3)"
    )
    table = lake.load_table(("tpch", "t1"))
    assert table.location() == location
    assert table.format_version == 1
    assert table.properties["write.parquet.compression-codec"] == "zstd"
    assert table.properties["flag"] == "true"
    assert table.properties["n"] == "-3"

    location = f"file://{tmp_path / 'other'}"
    ctx.execute(f"CREATE TABLE t2 (x INT) LOCATION '{location}' WITH ('k' = 'v')")
    table = lake.load_table(("tpch", "t2"))
    assert table.location() == location
    assert table.properties["k"] == "v"


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("statement", "error"),
    [
        (
            "CREATE OR REPLACE TABLE t (x INT)",
            "`CREATE OR REPLACE TABLE` is not supported for catalog tables",
        ),
        (
            "CREATE TABLE t (x INT) AS SELECT 1 AS x",
            r"`CREATE TABLE \.\.\. AS SELECT` does not take column definitions",
        ),
        (
            "CREATE TABLE t PARTITION BY (x) AS SELECT 1 AS x",
            "`PARTITION BY` is not yet supported for catalog tables",
        ),
        (
            "CREATE TABLE t LIKE lineitem",
            r"`CREATE TABLE \.\.\. LIKE` is not supported for catalog tables",
        ),
        (
            "CREATE TEMP TABLE t LIKE lineitem",
            r"`CREATE TABLE \.\.\. LIKE` is not supported for catalog tables",
        ),
        (
            "CREATE TABLE t (x INT UNIQUE)",
            "only NOT NULL constraints are supported for catalog tables",
        ),
        (
            "CREATE TABLE t (x INT DEFAULT 1)",
            "only NOT NULL constraints are supported for catalog tables",
        ),
        (
            "CREATE TABLE t (x INT, PRIMARY KEY (x))",
            "only NOT NULL constraints are supported for catalog tables",
        ),
        ("CREATE TABLE t (x INT NULL NOT NULL)", "conflicting NULL/NOT NULL"),
        (
            "CREATE TABLE t (x INT) PARTITION BY (x)",
            "`PARTITION BY` is not yet supported for catalog tables; create the "
            "table unpartitioned",
        ),
        (
            "CREATE TABLE t (x INT) PARTITIONED BY (x)",
            "`PARTITIONED BY` is not yet supported for catalog tables",
        ),
        (
            "CREATE TABLE t (x INT) TBLPROPERTIES ('k' = 'v')",
            r"`TBLPROPERTIES` is not supported; use `WITH \(\.\.\.\)` instead",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('Location' = '/x')",
            "'Location' is not a table option; did you mean 'location'",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('FORMAT-VERSION' = 2)",
            "did you mean 'format-version'",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('uuid' = 'x')",
            "table option 'uuid' is reserved by Iceberg",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('k' = 1 + 1)",
            "table option values must be string, number or boolean literals",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('k' = NULL)",
            "table option values must be string, number or boolean literals",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('k' = 'a', 'k' = 'b')",
            "duplicate table option 'k'",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('' = 'a')",
            "table option names must not be empty",
        ),
        (
            "CREATE TABLE t (x INT) LOCATION '/a' WITH ('location' = '/b')",
            "`LOCATION` and table option 'location' cannot be combined",
        ),
        (
            "CREATE TABLE t (x INT) WITH ('location' = 'relative/path')",
            "location must be an absolute path or a URI",
        ),
    ],
)
def test_create_catalog_table_errors(
    lake: CountingCatalog, statement: str, error: str
) -> None:
    ctx = lake_ctx(lake)
    with pytest.raises(SQLInterfaceError, match=error):
        ctx.execute(statement)
    assert not lake.table_exists(("tpch", "t"))
    assert ctx.tables() == []


@pytest.mark.write_disk
def test_create_catalog_table_type_errors(lake: CountingCatalog) -> None:
    from polars.exceptions import DuplicateError

    ctx = lake_ctx(lake)
    with pytest.raises(DuplicateError, match="column 'x' is defined more than once"):
        ctx.execute("CREATE TABLE t (x INT, x BIGINT)")
    with pytest.raises(
        TypeError,
        match=r"CREATE TABLE: columns without a lossless Iceberg type: 'x' \(UInt64\)",
    ):
        ctx.execute("CREATE TABLE t (x UBIGINT)")
    with pytest.raises(TypeError, match="microsecond precision"):
        ctx.execute("CREATE TABLE t (x TIMESTAMP(9))")
    assert not lake.table_exists(("tpch", "t"))


@pytest.mark.write_disk
def test_create_catalog_table_missing_namespace(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NoSuchNamespaceError

    ctx = lake_ctx(lake)
    for name, namespace in (("sales.t", "sales"), ("TPCH.t", "TPCH")):
        with pytest.raises(
            NoSuchNamespaceError,
            match=(
                f"table '{name}' could not be created: namespace '{namespace}' "
                "does not exist in catalog 'lake'.*case-sensitive"
            ),
        ):
            ctx.execute(f"CREATE TABLE {name} (x INT)")

    # a catalog that raises another error for a missing namespace
    def create_table(*args: Any, **kwargs: Any) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(lake, "create_table", create_table)
    with pytest.raises(NoSuchNamespaceError, match=r"namespace 'sales'.*\(boom\)"):
        ctx.execute("CREATE TABLE sales.t (x INT)")
    with pytest.raises(RuntimeError, match="boom") as exc:
        ctx.execute("CREATE TABLE t (x INT)")
    assert exc.value.__notes__ == [
        "table 't' could not be created: catalog table 'tpch.t'"
    ]


def table_files(tmp_path: Path, *identifier: str) -> set[Path]:
    # the `lake` warehouse is `tmp_path`
    return set(tmp_path.joinpath(*identifier).rglob("*.parquet"))


@pytest.mark.write_disk
@pytest.mark.parametrize("name", ["lake.tpch.t", "tpch.t", "t"])
def test_create_catalog_table_as_select(lake: CountingCatalog, name: str) -> None:
    # #29892
    ctx = lake_ctx(lake)
    ctx.register("extra", pl.LazyFrame({"k": [1], "w": [0.5]}))
    res = ctx.execute(
        f"CREATE TABLE {name} AS SELECT l.k, l.v, CAST(l.k AS UTINYINT) AS u, e.w "
        "FROM lineitem AS l LEFT JOIN extra AS e ON l.k = e.k",
        eager=True,
    )
    assert res.item() == "CREATE TABLE lake.tpch.t"
    assert ctx.tables() == ["extra"]
    # columns are never required, and get lossless Iceberg types
    assert table_fields(lake, "tpch", "t") == [
        ("k", "long", False),
        ("v", "string", False),
        ("u", "int", False),
        ("w", "double", False),
    ]
    table = lake.load_table(("tpch", "t"))
    assert len(table.metadata.snapshots) == 1
    expected = pl.DataFrame(
        {"k": [1, 2], "v": ["a", "b"], "u": [1, 2], "w": [0.5, None]},
        schema_overrides={"u": pl.Int32},
    )
    res = ctx.execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, expected, check_row_order=False)
    assert_frame_equal(pl.scan_iceberg(table).collect(), res, check_row_order=False)


@pytest.mark.write_disk
def test_create_catalog_table_as_select_existing(
    lake: CountingCatalog, tmp_path: Path
) -> None:
    from pyiceberg.exceptions import TableAlreadyExistsError

    ctx = lake_ctx(lake)
    files = table_files(tmp_path, "tpch", "lineitem")
    # the query is planned, but not run: the cast would fail
    res = ctx.execute(
        "CREATE TABLE IF NOT EXISTS lineitem AS SELECT CAST(v AS INT) AS x FROM lineitem",
        eager=True,
    )
    assert res.item() == "CREATE TABLE lake.tpch.lineitem"
    with pytest.raises(NoSuchTableError):
        ctx.execute("CREATE TABLE IF NOT EXISTS lineitem AS SELECT * FROM nope")
    with pytest.raises(
        TableAlreadyExistsError,
        match=r"table 'lineitem' already exists: catalog table 'tpch\.lineitem'",
    ):
        ctx.execute("CREATE TABLE lineitem AS SELECT CAST(v AS INT) AS x FROM lineitem")
    res = ctx.execute("SELECT * FROM lineitem", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)
    assert table_files(tmp_path, "tpch", "lineitem") == files


@pytest.mark.write_disk
def test_create_catalog_table_as_select_created_meanwhile(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import TableAlreadyExistsError

    ctx = lake_ctx(lake)
    files = table_files(tmp_path, "tpch", "lineitem")
    metadata = set((tmp_path / "tpch" / "lineitem" / "metadata").iterdir())
    table_exists = lake.table_exists
    checks: list[Any] = []

    def first_check_misses(identifier: Any) -> bool:
        # the table appears after the existence check
        checks.append(identifier)
        return len(checks) > 1 and table_exists(identifier)

    monkeypatch.setattr(lake, "table_exists", first_check_misses)
    res = ctx.execute(
        "CREATE TABLE IF NOT EXISTS lineitem AS SELECT 1 AS x", eager=True
    )
    assert res.item() == "CREATE TABLE lake.tpch.lineitem"

    checks.clear()
    with pytest.raises(
        TableAlreadyExistsError, match="table 'lineitem' already exists"
    ):
        ctx.execute("CREATE TABLE lineitem AS SELECT 1 AS x")

    # the other writer's table is unchanged, and this write's files are deleted
    monkeypatch.setattr(lake, "table_exists", table_exists)
    res = ctx.execute("SELECT * FROM lineitem", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)
    assert len(lake.load_table(("tpch", "lineitem")).metadata.snapshots) == 1
    assert table_files(tmp_path, "tpch", "lineitem") == files
    assert set((tmp_path / "tpch" / "lineitem" / "metadata").iterdir()) == metadata


@pytest.mark.write_disk
def test_create_catalog_table_as_select_fails(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.exceptions import InvalidOperationError
    from polars.io.iceberg._sink import IcebergSinkState

    ctx = lake_ctx(lake)
    # a query that fails while it runs creates no table
    with pytest.raises(InvalidOperationError):
        ctx.execute("CREATE TABLE t AS SELECT CAST(v AS INT) AS x FROM lineitem")
    assert not lake.table_exists(("tpch", "t"))

    # a failed commit deletes the written files
    def fail(*args: Any, **kwargs: Any) -> None:
        assert table_files(tmp_path, "tpch", "t")
        msg = "commit failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(IcebergSinkState, "_commit_create", fail)
    with pytest.raises(RuntimeError, match="commit failed") as exc:
        ctx.execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__ == [
        "table 't' could not be created: catalog table 'tpch.t'"
    ]
    assert not lake.table_exists(("tpch", "t"))
    assert not table_files(tmp_path, "tpch", "t")

    # files under the object-storage layout are left, with a note
    with pytest.raises(RuntimeError, match="commit failed") as exc:
        ctx.execute(
            "CREATE TABLE t WITH ('write.object-storage.enabled' = true) "
            "AS SELECT * FROM lineitem"
        )
    assert exc.value.__notes__[0].startswith("data files may remain under '")


@pytest.mark.write_disk
def test_create_catalog_table_as_select_fails_location_suffix(
    lake: CountingCatalog, tmp_path: Path
) -> None:
    from polars.exceptions import InvalidOperationError

    # a fragment, query or glob in the location does not widen the cleanup
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep").touch()
    ctx = lake_ctx(lake)
    for suffix in ("#x", "?x", "*"):
        with pytest.raises(InvalidOperationError) as exc:
            ctx.execute(
                f"CREATE TABLE t LOCATION 'file://{other}{suffix}' "
                "AS SELECT CAST(v AS INT) AS x FROM lineitem"
            )
        assert (other / "keep").exists()
        assert exc.value.__notes__[0].startswith("data files may remain under '")


@pytest.mark.write_disk
def test_create_catalog_table_as_select_lost_commit_response(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException
    from pyiceberg.table import CreateTableTransaction

    commit_transaction = CreateTableTransaction.commit_transaction

    def commit_then_fail(self: Any) -> Any:
        commit_transaction(self)
        msg = "response lost"
        raise CommitStateUnknownException(msg)

    monkeypatch.setattr(CreateTableTransaction, "commit_transaction", commit_then_fail)
    res = lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.t"
    res = lake_ctx(lake).execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)


@pytest.mark.write_disk
def test_create_catalog_table_as_select_lost_response_snapshot_expired(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException
    from pyiceberg.table import CreateTableTransaction

    commit_transaction = CreateTableTransaction.commit_transaction
    extra = pl.DataFrame({"k": [3], "v": ["c"]})

    def commit_then_fail(self: Any) -> Any:
        commit_transaction(self)
        # another writer appends and expires the create's snapshot
        table = lake.load_table(("tpch", "t"))
        first = table.current_snapshot().snapshot_id
        extra.write_iceberg(table, mode="append")
        table.maintenance.expire_snapshots().by_id(first).commit()
        msg = "response lost"
        raise CommitStateUnknownException(msg)

    monkeypatch.setattr(CreateTableTransaction, "commit_transaction", commit_then_fail)
    res = lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.t"
    res = lake_ctx(lake).execute("SELECT * FROM t", eager=True)
    expected = pl.concat([LINEITEM, extra])
    assert_frame_equal(res, expected, check_row_order=False)


def test_create_catalog_table_as_select_not_serializable(
    lake: CountingCatalog,
) -> None:
    from polars.exceptions import ComputeError
    from polars.io.iceberg._sink import IcebergSinkState

    # elsewhere, the staged table would be loaded by name: maybe another writer's
    transaction = lake.create_table_transaction(
        ("tpch", "t"), schema=LINEITEM.to_arrow().schema
    )
    state = IcebergSinkState.new_create(transaction)
    with pytest.raises(TypeError, match="staged Iceberg create cannot be serialized"):
        pickle.dumps(state)
    lf = state.attach_sink(LINEITEM.lazy())
    with pytest.raises(ComputeError, match="staged Iceberg create cannot be"):
        lf.serialize()


@pytest.mark.write_disk
def test_create_catalog_table_as_select_unknown_outcome(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException
    from pyiceberg.table import CreateTableTransaction

    commit_transaction = CreateTableTransaction.commit_transaction
    load_table = lake.load_table
    committed: list[bool] = []

    def commit_then_fail(self: Any) -> Any:
        commit_transaction(self)
        committed.append(True)
        msg = "response lost"
        raise CommitStateUnknownException(msg)

    def load_fails_for_t(identifier: Any) -> Any:
        # the catalog becomes unreachable after the commit
        if committed and tuple(identifier) == ("tpch", "t"):
            msg = "catalog unreachable"
            raise ConnectionError(msg)
        return load_table(identifier)

    monkeypatch.setattr(CreateTableTransaction, "commit_transaction", commit_then_fail)
    monkeypatch.setattr(lake, "load_table", load_fails_for_t)
    # the create landed, so its data files must be kept
    with pytest.raises(CommitStateUnknownException, match="could not verify") as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__[0].startswith(
        "the create may have landed; data files kept under '"
    )
    assert exc.value.__notes__[1] == (
        "table 't' may not have been created: catalog table 'tpch.t'"
    )

    monkeypatch.setattr(lake, "load_table", load_table)
    res = lake_ctx(lake).execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)


@pytest.mark.write_disk
def test_create_catalog_table_as_select_unknown_create_not_landed(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException
    from pyiceberg.table import CreateTableTransaction

    def unknown(self: Any) -> Any:
        msg = "response lost"
        raise CommitStateUnknownException(msg)

    monkeypatch.setattr(CreateTableTransaction, "commit_transaction", unknown)
    # the create may still land later, so its data files are kept
    with pytest.raises(CommitStateUnknownException, match="response lost") as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__[0].startswith(
        "the create may have landed; data files kept under '"
    )
    assert not lake.table_exists(("tpch", "t"))
    assert table_files(tmp_path, "tpch", "t")


@pytest.mark.write_disk
def test_create_catalog_table_as_select_rejected_unverified(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import TableAlreadyExistsError
    from pyiceberg.table import CreateTableTransaction

    load_table = lake.load_table
    rejected: list[bool] = []

    def other_writer_then_rejected(self: Any) -> Any:
        lake.create_table(("tpch", "t"), schema=LINEITEM.to_arrow().schema)
        rejected.append(True)
        msg = "created meanwhile"
        raise TableAlreadyExistsError(msg)

    def load_fails_for_t(identifier: Any) -> Any:
        if rejected and tuple(identifier) == ("tpch", "t"):
            msg = "catalog unreachable"
            raise ConnectionError(msg)
        return load_table(identifier)

    monkeypatch.setattr(
        CreateTableTransaction, "commit_transaction", other_writer_then_rejected
    )
    monkeypatch.setattr(lake, "load_table", load_fails_for_t)
    # a rejection stands when it cannot be verified; the files are kept
    res = lake_ctx(lake).execute(
        "CREATE TABLE IF NOT EXISTS t AS SELECT * FROM lineitem", eager=True
    )
    assert res.item() == "CREATE TABLE lake.tpch.t"
    monkeypatch.setattr(lake, "load_table", load_table)
    assert not lake.load_table(("tpch", "t")).metadata.snapshots
    assert table_files(tmp_path, "tpch", "t")


@pytest.mark.write_disk
@pytest.mark.parametrize("staged", [True, False])
def test_create_catalog_table_as_select_commit_not_rejected(
    lake: CountingCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staged: bool,
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException

    if not staged:
        monkeypatch.setattr(lake, "create_table_transaction", no_staged_create)

    def connection_lost(*args: Any, **kwargs: Any) -> None:
        msg = "connection lost"
        raise ConnectionError(msg)

    monkeypatch.setattr(lake, "commit_table", connection_lost)
    # a commit that was not rejected may still land, so its data files are kept
    with pytest.raises(CommitStateUnknownException, match="may still land") as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert isinstance(exc.value.__cause__, ConnectionError)
    assert "landed; data files kept under '" in exc.value.__notes__[0]
    assert table_files(tmp_path, "tpch", "t")


@pytest.mark.write_disk
def test_create_catalog_table_as_select_cleanup_storage_options(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.iceberg import _sink

    def fail(*args: Any, **kwargs: Any) -> None:
        msg = "commit failed"
        raise RuntimeError(msg)

    deleted_with: list[dict[str, Any]] = []
    delete_dir = _sink._delete_dir

    def record(io: Any, location: str) -> None:
        deleted_with.append(dict(io.properties))
        delete_dir(io, location)

    monkeypatch.setattr(_sink.IcebergSinkState, "_commit_create", fail)
    monkeypatch.setattr(_sink, "_delete_dir", record)
    # the files are deleted with the storage options they were written with
    ctx = lake_ctx(lake, storage_options={"polars.test-option": "x"})
    with pytest.raises(RuntimeError, match="commit failed"):
        ctx.execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert deleted_with
    assert deleted_with[0]["polars.test-option"] == "x"

    with pytest.raises(NotImplementedError):
        delete_dir(object(), "file:///nowhere")  # type: ignore[arg-type]


@pytest.mark.write_disk
def test_create_catalog_table_as_select_cleanup_fails(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.iceberg import _sink

    def fail(*args: Any, **kwargs: Any) -> None:
        msg = "commit failed"
        raise RuntimeError(msg)

    def delete_fails(*args: Any, **kwargs: Any) -> None:
        msg = "permission denied"
        raise PermissionError(msg)

    monkeypatch.setattr(_sink.IcebergSinkState, "_commit_create", fail)
    monkeypatch.setattr(_sink, "_delete_dir", delete_fails)
    with pytest.raises(RuntimeError, match="commit failed") as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__[0].startswith("data files may remain under '")
    assert "/tpch/t/data/" in exc.value.__notes__[0]


@pytest.mark.write_disk
def test_create_catalog_table_as_select_fsspec_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fsspec")
    from polars.io.iceberg._sink import IcebergSinkState

    catalog = SqlCatalog(
        "lake",
        uri=f"sqlite:///{tmp_path / 'fs.db'}",
        warehouse=f"file://{tmp_path}",
        **{"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"},
    )
    catalog.create_namespace("ns")
    ctx = pl.SQLContext().register_catalog(
        "lake", catalog, default_namespace="ns", default=True
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        assert table_files(tmp_path, "ns", "t")
        msg = "commit failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(IcebergSinkState, "_commit_create", fail)
    with pytest.raises(RuntimeError, match="commit failed"):
        ctx.execute("CREATE TABLE t AS SELECT 1 AS x")
    assert not table_files(tmp_path, "ns", "t")


def no_staged_create(*args: Any, **kwargs: Any) -> None:
    from pyiceberg.exceptions import BadRequestError

    msg = "staged create rejected"
    raise BadRequestError(msg)


@pytest.mark.write_disk
@pytest.mark.parametrize("rejected", ["400", "501"])
def test_create_catalog_table_as_select_without_staged_create(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch, rejected: str
) -> None:
    from pyiceberg.exceptions import TableAlreadyExistsError

    def not_implemented(*args: Any, **kwargs: Any) -> None:
        msg = "staged create not implemented"
        raise NotImplementedError(msg)

    # some REST catalogs reject staged creates: the table is created, then filled
    rejection = no_staged_create if rejected == "400" else not_implemented
    monkeypatch.setattr(lake, "create_table_transaction", rejection)
    ctx = lake_ctx(lake)
    res = ctx.execute("CREATE TABLE t AS SELECT * FROM lineitem", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.t"
    assert len(lake.load_table(("tpch", "t")).metadata.snapshots) == 1
    res = ctx.execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)

    res = ctx.execute("CREATE TABLE IF NOT EXISTS t AS SELECT 1 AS x", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.t"
    assert len(lake.load_table(("tpch", "t")).metadata.snapshots) == 1

    # the table appears after the existence check
    table_exists = lake.table_exists
    monkeypatch.setattr(lake, "table_exists", lambda identifier: False)
    res = ctx.execute("CREATE TABLE IF NOT EXISTS t AS SELECT 1 AS x", eager=True)
    assert res.item() == "CREATE TABLE lake.tpch.t"
    with pytest.raises(TableAlreadyExistsError, match="table 't' already exists"):
        ctx.execute("CREATE TABLE t AS SELECT 1 AS x")
    monkeypatch.setattr(lake, "table_exists", table_exists)
    res = ctx.execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)


@pytest.mark.write_disk
def test_create_catalog_table_as_select_without_staged_create_fails(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitFailedException

    from polars.exceptions import InvalidOperationError
    from polars.io.iceberg._sink import IcebergSinkState

    monkeypatch.setattr(lake, "create_table_transaction", no_staged_create)
    ctx = lake_ctx(lake)
    # the table created for a failing query is dropped again
    with pytest.raises(InvalidOperationError) as exc:
        ctx.execute("CREATE TABLE t AS SELECT CAST(v AS INT) AS x FROM lineitem")
    assert exc.value.__notes__ == [
        "table 't' could not be created: catalog table 'tpch.t'"
    ]
    assert not lake.table_exists(("tpch", "t"))
    assert not table_files(tmp_path, "tpch", "t")

    # a failed commit is not taken for a table created meanwhile
    def fail(*args: Any, **kwargs: Any) -> None:
        assert table_files(tmp_path, "tpch", "t")
        msg = "commit failed"
        raise CommitFailedException(msg)

    monkeypatch.setattr(IcebergSinkState, "_commit", fail)
    with pytest.raises(CommitFailedException, match="commit failed") as exc_commit:
        ctx.execute("CREATE TABLE IF NOT EXISTS t AS SELECT * FROM lineitem")
    assert exc_commit.value.__notes__ == [
        "table 't' could not be created: catalog table 'tpch.t'"
    ]
    assert not lake.table_exists(("tpch", "t"))
    assert not table_files(tmp_path, "tpch", "t")

    # files under the object-storage layout are left, with a note
    with pytest.raises(CommitFailedException) as exc_commit:
        ctx.execute(
            "CREATE TABLE t WITH ('write.object-storage.enabled' = true) "
            "AS SELECT * FROM lineitem"
        )
    assert exc_commit.value.__notes__[0].startswith("data files may remain under '")
    assert exc_commit.value.__notes__[1:] == [
        "table 't' could not be created: catalog table 'tpch.t'"
    ]
    assert not lake.table_exists(("tpch", "t"))


@pytest.mark.write_disk
def test_create_catalog_table_as_select_without_staged_create_written_meanwhile(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.iceberg._sink import IcebergSinkState

    monkeypatch.setattr(lake, "create_table_transaction", no_staged_create)

    def write_then_fail(*args: Any, **kwargs: Any) -> None:
        # another writer fills the new table first
        pl.DataFrame({"k": [3], "v": ["c"]}).write_iceberg(
            lake.load_table(("tpch", "t")), mode="append"
        )
        msg = "commit failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(IcebergSinkState, "_commit", write_then_fail)
    with pytest.raises(RuntimeError, match="commit failed") as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__ == [
        "table 't' could not be filled: catalog table 'tpch.t' was left in the catalog"
    ]
    monkeypatch.undo()
    res = lake_ctx(lake).execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, pl.DataFrame({"k": [3], "v": ["c"]}))


@pytest.mark.write_disk
def test_create_catalog_table_as_select_without_staged_create_unknown_outcome(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException

    from polars.io.iceberg._sink import IcebergSinkState

    monkeypatch.setattr(lake, "create_table_transaction", no_staged_create)

    def unknown(*args: Any, **kwargs: Any) -> None:
        msg = "response lost"
        raise CommitStateUnknownException(msg)

    monkeypatch.setattr(IcebergSinkState, "_commit", unknown)
    # the rows may have landed, so the table and its files are kept
    with pytest.raises(CommitStateUnknownException) as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__[0].startswith(
        "the rows may have landed; data files kept under '"
    )
    assert exc.value.__notes__[1] == (
        "table 't' was created, its rows may be missing: catalog table 'tpch.t'"
    )
    assert lake.table_exists(("tpch", "t"))


@pytest.mark.write_disk
def test_create_catalog_table_as_select_lost_response_created_meanwhile(
    lake: CountingCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import CommitStateUnknownException
    from pyiceberg.table import CreateTableTransaction

    def other_writer_then_unknown(self: Any) -> Any:
        # another writer creates the table, and this commit's response is lost
        lake.create_table(("tpch", "t"), schema=LINEITEM.to_arrow().schema)
        msg = "response lost"
        raise CommitStateUnknownException(msg)

    monkeypatch.setattr(
        CreateTableTransaction, "commit_transaction", other_writer_then_unknown
    )
    # this create may have landed and been renamed: its files are kept
    with pytest.raises(CommitStateUnknownException, match="response lost") as exc:
        lake_ctx(lake).execute("CREATE TABLE IF NOT EXISTS t AS SELECT * FROM lineitem")
    assert exc.value.__notes__[0].startswith(
        "the create may have landed; data files kept under '"
    )
    assert not lake.load_table(("tpch", "t")).metadata.snapshots
    assert table_files(tmp_path, "tpch", "t")


@pytest.mark.write_disk
@pytest.mark.parametrize("staged", [True, False])
def test_create_catalog_table_as_select_fails_after_commit(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch, staged: bool
) -> None:
    from polars.io.iceberg._sink import IcebergSinkState

    if not staged:
        monkeypatch.setattr(lake, "create_table_transaction", no_staged_create)
    commit = IcebergSinkState._commit

    def commit_then_fail(self: Any, *args: Any, **kwargs: Any) -> None:
        commit(self, *args, **kwargs)
        msg = "failed after the commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(IcebergSinkState, "_commit", commit_then_fail)
    # the data files of a landed commit are kept
    with pytest.raises(RuntimeError, match="failed after the commit") as exc:
        lake_ctx(lake).execute("CREATE TABLE t AS SELECT * FROM lineitem")
    assert exc.value.__notes__[0].startswith("data files may remain under '")
    monkeypatch.undo()
    res = lake_ctx(lake).execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)


@pytest.mark.write_disk
def test_create_catalog_table_as_select_query_text(lake: CountingCatalog) -> None:
    # the query is passed on as SQL text
    ctx = lake_ctx(lake)
    ctx.execute(
        "CREATE TABLE t AS WITH c AS (SELECT k, 'it''s' AS s FROM lake.tpch.lineitem) "
        'SELECT k::DOUBLE AS "k value", s FROM c WHERE k > 1'
    )
    res = ctx.execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, pl.DataFrame({"k value": [2.0], "s": ["it's"]}))


@pytest.mark.write_disk
def test_create_catalog_table_as_select_options(
    lake: CountingCatalog, tmp_path: Path
) -> None:
    from pyiceberg.exceptions import NoSuchNamespaceError

    ctx = lake_ctx(lake)
    location = f"file://{tmp_path / 'custom'}"
    ctx.execute(
        f"CREATE TABLE t WITH ('location' = '{location}', 'format-version' = 1, "
        "'k' = 'v') AS SELECT * FROM lineitem"
    )
    table = lake.load_table(("tpch", "t"))
    assert table.location() == location
    assert table.format_version == 1
    assert table.properties["k"] == "v"
    assert list((tmp_path / "custom").rglob("*.parquet"))

    location = f"file://{tmp_path / 'clause'}"
    ctx.execute(f"CREATE TABLE t3 LOCATION '{location}' AS SELECT * FROM lineitem")
    assert lake.load_table(("tpch", "t3")).location() == location
    assert list((tmp_path / "clause").rglob("*.parquet"))

    with pytest.raises(
        NoSuchNamespaceError,
        match=r"table 'sales\.t' could not be created: namespace 'sales' does not exist",
    ):
        ctx.execute("CREATE TABLE sales.t AS SELECT 1 AS x")
    with pytest.raises(TypeError, match="columns without a lossless Iceberg type"):
        ctx.execute("CREATE TABLE t2 AS SELECT CAST(k AS UBIGINT) AS x FROM lineitem")
    assert not lake.table_exists(("tpch", "t2"))


@pytest.mark.write_disk
def test_create_catalog_table_without_warehouse(
    lake: CountingCatalog, tmp_path: Path
) -> None:
    from pyiceberg.exceptions import TableAlreadyExistsError

    catalog = SqlCatalog("bare", uri=f"sqlite:///{tmp_path / 'bare.db'}")
    catalog.create_namespace("ns")
    ctx = pl.SQLContext().register_catalog("bare", catalog, default=True)
    # the catalog's error, with the table named in a note
    with pytest.raises(ValueError) as exc:
        ctx.execute("CREATE TABLE ns.t (x INT)")
    assert exc.value.__notes__ == [
        "table 'ns.t' could not be created: catalog table 'ns.t'"
    ]
    location = f"file://{tmp_path / 't'}"
    ctx.execute(f"CREATE TABLE ns.t (x INT) WITH ('location' = '{location}')")
    assert catalog.load_table(("ns", "t")).location() == location
    # an existing table, though the catalog would first report the missing location
    with pytest.raises(TableAlreadyExistsError, match=r"table 'ns\.t' already exists"):
        ctx.execute("CREATE TABLE ns.t (x INT)")


@pytest.mark.write_disk
def test_create_catalog_table_from_catalog_config(lake: CountingCatalog) -> None:
    from polars.io.iceberg import IcebergCatalogConfig

    config = IcebergCatalogConfig.from_catalog(lake)
    ctx = pl.SQLContext().register_catalog(
        "lake", config, default_namespace="tpch", default=True
    )
    ctx.execute("CREATE TABLE t (x INT NOT NULL)")
    assert table_fields(lake, "tpch", "t") == [("x", "int", True)]


@pytest.mark.write_disk
def test_create_catalog_table_hidden_by_registered_table(
    lake: CountingCatalog,
) -> None:
    ctx = lake_ctx(lake)
    ctx.register_many({"t": pl.LazyFrame({"x": [1]}), "lake.tpch.u": pl.LazyFrame()})
    for statement, name in (
        ("CREATE TABLE t (x INT)", "t"),
        ("CREATE TABLE T (x INT)", "T"),
        ("CREATE TABLE IF NOT EXISTS t (x INT)", "t"),
    ):
        with pytest.raises(
            SQLInterfaceError,
            match=(
                rf"table '{name}' would be hidden by registered table 't'; write the "
                rf"full name 'lake\.tpch\.{name}', or unregister it"
            ),
        ):
            ctx.execute(statement)
    for name in ("lake.tpch.u", "u"):
        with pytest.raises(
            SQLInterfaceError,
            match=rf"table '{name}' would be hidden by registered table "
            r"'lake\.tpch\.u'; unregister it$",
        ):
            ctx.execute(f"CREATE TABLE {name} (x INT)")
    assert not lake.table_exists(("tpch", "u"))
    assert not lake.table_exists(("tpch", "t"))

    # the full name creates the catalog table, which the registered table still hides
    ctx.execute("CREATE TABLE lake.tpch.t (y INT)")
    assert ctx.execute("SELECT * FROM t", eager=True).columns == ["x"]
    assert ctx.execute("SELECT * FROM lake.tpch.t", eager=True).columns == ["y"]


@pytest.mark.write_disk
def test_create_table_without_namespace(lake: CountingCatalog) -> None:
    # a name the default catalog applies to doesn't silently create a registered table
    ctx = lake_ctx(lake, default_namespace=None)
    for statement in ("CREATE TABLE t (x INT)", "CREATE TABLE t AS SELECT 1 AS x"):
        with pytest.raises(
            SQLInterfaceError,
            match=(
                r"table 't' has no namespace: catalog 'lake' has no default "
                r"namespace; write 'lake\.<namespace>\.t', or use `CREATE TEMP TABLE`"
            ),
        ):
            ctx.execute(statement)
    with pytest.raises(SQLInterfaceError, match=r"write 'lake\.<namespace>\.t'$"):
        ctx.execute("CREATE TABLE lake.t (x INT)")
    ctx.execute("CREATE TEMP TABLE t AS SELECT 1 AS x")
    ctx.execute("CREATE TABLE tpch.t2 (x INT)")
    assert ctx.tables() == ["t"]
    assert lake.table_exists(("tpch", "t2"))

    # a name that no catalog applies to creates a registered table
    ctx = lake_ctx(lake, default=False)
    ctx.execute("CREATE TABLE tpch.t AS SELECT 1 AS x")
    assert ctx.execute("SELECT x FROM tpch.t", eager=True).item() == 1
    assert not lake.table_exists(("tpch", "t"))


@pytest.mark.write_disk
def test_create_temp_table_with_catalogs(lake: CountingCatalog) -> None:
    # a temporary table is a registered table, also where the default catalog applies
    ctx = lake_ctx(lake)
    for name in ("t", "tpch.t", "lineitem"):
        ctx.execute(f"CREATE TEMP TABLE {name} AS SELECT 1 AS x")
    assert sorted(ctx.tables()) == ["lineitem", "t", "tpch.t"]
    assert ctx.execute("SELECT x FROM lineitem", eager=True).item() == 1
    assert not lake.table_exists(("tpch", "t"))
    res = ctx.execute("SELECT * FROM lake.tpch.lineitem", eager=True)
    assert_frame_equal(res, LINEITEM, check_row_order=False)

    for name in ("lake.tpch.t2", "LAKE.t2"):
        with pytest.raises(
            SQLInterfaceError,
            match="temporary tables cannot be created in catalog 'lake'",
        ):
            ctx.execute(f"CREATE TEMP TABLE {name} AS SELECT 1 AS x")


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
