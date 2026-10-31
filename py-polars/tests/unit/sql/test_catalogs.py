from __future__ import annotations

import io
import pickle
import re
import shutil
from datetime import UTC, date
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
        "DELETE FROM lineitem WHERE k = 1",
        "DELETE FROM lake.tpch.lineitem",
    ],
)
def test_catalog_statement_is_not_supported(
    lake: CountingCatalog, statement: str
) -> None:
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
            "CREATE TABLE t (x INT) PARTITION BY RANGE (x)",
            r"unsupported partition field `RANGE\(x\)`; expected a column, or year",
        ),
        ("CREATE TABLE t (x INT) PARTITION BY (x + 1)", "unsupported partition field"),
        (
            "CREATE TABLE t (x INT) PARTITION BY (bucket(4))",
            "unsupported partition field",
        ),
        (
            "CREATE TABLE t (s STRUCT<a INT>) PARTITION BY (s.a)",
            r"`PARTITION BY` takes unqualified column names; found `s\.a`; "
            "partitioning by a nested field is not yet supported",
        ),
        (
            "CREATE TABLE t (s STRUCT<a INT>) PARTITION BY (bucket(4, s.a))",
            r"`PARTITION BY` takes unqualified column names; found `s\.a`",
        ),
        (
            "CREATE TABLE t (f INT) PARTITION BY (t.f)",
            r"`PARTITION BY` takes unqualified column names; found `t\.f`",
        ),
        (
            "CREATE TABLE t (s STRUCT<a INT>) PARTITION BY (bucket(s.a, 4))",
            r"`bucket\(s\.a, 4\)` takes the number first, as in `bucket\(4, s\.a\)`",
        ),
        (
            "CREATE TABLE t (s STRUCT<a INT>) PARTITION BY ((s).a)",
            r"unsupported partition field `\(s\)\.a`",
        ),
        (
            "CREATE TABLE t (d DATE) PARTITION BY (date(d))",
            r"unsupported partition field `date\(d\)`",
        ),
        (
            "CREATE TABLE t (ts TIMESTAMP) PARTITION BY (date_hour(ts))",
            r"unsupported partition field `date_hour\(ts\)`",
        ),
        (
            "CREATE TABLE t (x INT) PARTITION BY (bucket(x, 4))",
            r"`bucket\(x, 4\)` takes the number first, as in `bucket\(4, x\)`",
        ),
        (
            "CREATE TABLE t (x INT) PARTITION BY (bucket(0, x))",
            r"the first argument of `bucket\(0, x\)` must be an integer from 1 to "
            "2147483647",
        ),
        (
            "CREATE TABLE t (x INT) PARTITION BY (truncate(-2, x))",
            "must be an integer from 1 to 2147483647",
        ),
        (
            "CREATE TABLE t (x INT) PARTITION BY (bucket(2147483648, x))",
            "must be an integer from 1 to 2147483647",
        ),
        ("CREATE TABLE t (x INT) PARTITION BY (x, x)", "duplicate partition field `x`"),
        (
            "CREATE TABLE t (d DATE) PARTITION BY (year(d), days(d))",
            r"partition fields `year\(d\)` and `days\(d\)` both partition column 'd' "
            "by time",
        ),
        (
            "CREATE TABLE t (x INT) PARTITIONED BY (x INT)",
            "`PARTITIONED BY` takes column names only; found `x INT`; types and "
            "constraints belong in the column list",
        ),
        (
            "CREATE TABLE t (x INT) PARTITIONED BY (x) PARTITION BY (x)",
            "`PARTITION BY` and `PARTITIONED BY` cannot be combined",
        ),
        (
            "CREATE TABLE t (x INT) PARTITION BY (y)",
            "partition column 'y' is not a column of table 't'",
        ),
        (
            'CREATE TABLE t (x INT) PARTITION BY ("X")',
            r"partition column 'X' is not a column of table 't'; "
            r"valid columns: \['x'\]",
        ),
        (
            "CREATE TABLE t PARTITION BY (y) AS SELECT 1 AS x, 2 AS z",
            r"partition column 'y' is not a column of table 't'; "
            r"valid columns: \['x', 'z'\]",
        ),
        (
            "CREATE TABLE t (x INT) PARTITION BY (day(x))",
            "cannot partition table 't' by day of column 'x' with type int",
        ),
        (
            "CREATE TABLE t (b BOOLEAN) PARTITION BY (bucket(4, b))",
            r"by bucket\[4\] of column 'b' with type boolean",
        ),
        (
            "CREATE TABLE t (s STRUCT<a INT>) PARTITION BY (s)",
            "by identity of column 's' with type struct",
        ),
        (
            "CREATE TABLE t (d DATE, d_day INT) PARTITION BY (day(d))",
            "partition field 'd_day' of table 't' would have the name of a column",
        ),
        (
            'CREATE TABLE t ("s.f" DATE, s STRUCT<f_day INT>) PARTITION BY (day("s.f"))',
            r"partition field 's\.f_day' of table 't' would have the name of a column",
        ),
        (
            "CREATE TEMP TABLE t (x INT) PARTITION BY (x)",
            "`PARTITION BY` is supported only for catalog tables",
        ),
        (
            "CREATE TEMP TABLE t (x INT) PARTITIONED BY (x)",
            "`PARTITIONED BY` is supported only for catalog tables",
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


def partition_fields(catalog: CountingCatalog, *identifier: str) -> list[Any]:
    table = catalog.load_table(identifier)
    schema = table.schema()
    return [
        (f.name, str(f.transform), schema.find_column_name(f.source_id))
        for f in table.spec().fields
    ]


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        (
            "PARTITION BY (p, bucket(4, id), truncate(2, x), day(ts))",
            [
                ("p", "identity", "p"),
                ("id_bucket_4", "bucket[4]", "id"),
                ("x_trunc_2", "truncate[2]", "x"),
                ("ts_day", "day", "ts"),
            ],
        ),
        ("PARTITION BY p", [("p", "identity", "p")]),
        ("PARTITIONED BY (p, x)", [("p", "identity", "p"), ("x", "identity", "x")]),
        (
            "PARTITION BY (YEARS(d), Bucket(8, id), bucket(4, id), hours(ts))",
            [
                ("d_year", "year", "d"),
                ("id_bucket_8", "bucket[8]", "id"),
                ("id_bucket_4", "bucket[4]", "id"),
                ("ts_hour", "hour", "ts"),
            ],
        ),
        (
            "PARTITION BY (month(d), hour(ts))",
            [("d_month", "month", "d"), ("ts_hour", "hour", "ts")],
        ),
        ("PARTITION BY (days(d))", [("d_day", "day", "d")]),
        (
            "WITH ('format-version' = 1) PARTITION BY (bucket(4, id), day(ts))",
            [("id_bucket_4", "bucket[4]", "id"), ("ts_day", "day", "ts")],
        ),
        (
            "PARTITION BY ((p), day((ts)), bucket(4, (id)))",
            [
                ("p", "identity", "p"),
                ("ts_day", "day", "ts"),
                ("id_bucket_4", "bucket[4]", "id"),
            ],
        ),
        # a quoted name with a dot is a top-level column
        ('PARTITION BY ("s.f")', [("s.f", "identity", "s.f")]),
        ('PARTITION BY (bucket(4, "s.f"))', [("s.f_bucket_4", "bucket[4]", "s.f")]),
    ],
)
def test_create_partitioned_catalog_table(
    lake: CountingCatalog, clause: str, expected: list[Any]
) -> None:
    ctx = lake_ctx(lake)
    # a struct first: the create renumbers field ids
    ctx.execute(
        "CREATE TABLE t (s STRUCT<a INT, b INT>, id BIGINT, p VARCHAR, x VARCHAR, "
        'd DATE, ts TIMESTAMP, "s.f" INT) ' + clause
    )
    assert partition_fields(lake, "tpch", "t") == expected
    spec = lake.load_table(("tpch", "t")).spec()
    assert [f.field_id for f in spec.fields] == list(range(1000, 1000 + len(expected)))


@pytest.mark.write_disk
@pytest.mark.parametrize("staged", [True, False])
def test_create_partitioned_catalog_table_as_select(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch, staged: bool
) -> None:
    if not staged:
        monkeypatch.setattr(lake, "create_table_transaction", no_staged_create)
    ctx = lake_ctx(lake)
    src = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "p": ["a", "b", "a", None],
            "d": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 2, 1), None],
        }
    )
    ctx.register("src", src)
    ctx.execute("CREATE TABLE t PARTITION BY (p, month(d)) AS SELECT * FROM src")
    assert partition_fields(lake, "tpch", "t") == [
        ("p", "identity", "p"),
        ("d_month", "month", "d"),
    ]
    table = lake.load_table(("tpch", "t"))
    assert len(table.metadata.snapshots) == 1
    # months since 1970
    partitions = table.inspect.files().column("partition").to_pylist()
    assert sorted(partitions, key=str) == sorted(
        [
            {"p": "a", "d_month": 648},
            {"p": "b", "d_month": 648},
            {"p": "a", "d_month": 649},
            {"p": None, "d_month": None},
        ],
        key=str,
    )
    res = ctx.execute("SELECT * FROM t", eager=True)
    assert_frame_equal(res, src, check_row_order=False)


@pytest.mark.write_disk
def test_create_partitioned_catalog_table_checks(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    with pytest.raises(
        NotImplementedError,
        match=r"partitioning by truncate\[2\] of a decimal column \('x'\)",
    ):
        ctx.execute("CREATE TABLE t (x DECIMAL(10, 2)) PARTITION BY (truncate(2, x))")
    assert not lake.table_exists(("tpch", "t"))

    # checked also when the table exists, and before the query runs: the cast fails
    error = "partition column 'x' is not a column of table 'lineitem'"
    with pytest.raises(SQLInterfaceError, match=error):
        ctx.execute("CREATE TABLE IF NOT EXISTS lineitem (k INT) PARTITION BY (x)")
    with pytest.raises(SQLInterfaceError, match="by day of column 'k' with type int"):
        ctx.execute(
            "CREATE TABLE t PARTITION BY (day(k)) AS "
            "SELECT CAST(v AS INT) AS k FROM lineitem"
        )
    assert not lake.table_exists(("tpch", "t"))

    # a partition name that is a nested field's
    ctx.register("src", pl.DataFrame({"s.f": [date(2024, 1, 1)], "s": [{"f_day": 1}]}))
    error = r"partition field 's\.f_day' of table '(t|lineitem)' would have the name"
    with pytest.raises(SQLInterfaceError, match=error):
        ctx.execute('CREATE TABLE t PARTITION BY (day("s.f")) AS SELECT * FROM src')
    with pytest.raises(SQLInterfaceError, match=error):
        ctx.execute(
            'CREATE TABLE IF NOT EXISTS lineitem PARTITION BY (day("s.f")) '
            "AS SELECT * FROM src"
        )
    assert not lake.table_exists(("tpch", "t"))


def test_create_partition_spec_duplicate_name() -> None:
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    from polars.sql.context import _create_partition_spec

    schema = Schema(NestedField(1, "id", LongType(), required=False))
    with pytest.raises(
        SQLInterfaceError,
        match="duplicate partition field name 'id_bucket_4' in table 't'",
    ):
        _create_partition_spec(schema, [("id", "bucket[4]"), ("id", "bucket[4]")], "t")


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
def test_if_exists_acts_on_registered_tables_first(lake: CountingCatalog) -> None:
    # a cleanup script keeps working once a default catalog is attached
    ctx = lake_ctx(lake)
    ctx.register("tmp", pl.LazyFrame({"x": [1]}))
    ctx.register("staged", pl.LazyFrame({"x": [1]}))
    for statement in (
        "DROP TABLE IF EXISTS tmp",
        "DROP TABLE IF EXISTS tmp",
        "TRUNCATE TABLE IF EXISTS staged",
        "TRUNCATE TABLE IF EXISTS tpch.missing",
    ):
        ctx.execute(statement)
    assert ctx.tables() == ["staged"]
    assert ctx.execute("SELECT * FROM staged", eager=True).is_empty()
    assert lake.loads == [("tpch", "missing")]  # looked up, a no-op
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
def test_drop_catalog_table_purge_refused(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    purges: list[Any] = []

    def refuse(identifier: Any) -> None:
        purges.append(identifier)
        msg = "purge is not supported"
        raise RuntimeError(msg)

    # a catalog that refuses any purge, before it looks for the table
    ctx = lake_ctx(lake)
    monkeypatch.setattr(lake, "purge_table", refuse)
    for if_exists in ["", "IF EXISTS "]:
        with pytest.raises(RuntimeError, match="purge is not supported") as exc:
            ctx.execute(f"DROP TABLE {if_exists}orders PURGE")
        assert exc.value.__notes__ == [
            "dropping table 'orders' failed: catalog table 'tpch.orders' of "
            "catalog 'lake'"
        ]
    assert lake.table_exists(("tpch", "orders"))

    purges.clear()
    ctx.execute("DROP TABLE IF EXISTS nope PURGE")
    with pytest.raises(NoSuchTableError, match="table 'nope' does not exist"):
        ctx.execute("DROP TABLE nope PURGE")
    assert purges == []

    # a table whose existence can't be checked is not taken for a missing one
    def unknown(identifier: Any) -> bool:
        msg = "not allowed"
        raise PermissionError(msg)

    monkeypatch.setattr(lake, "table_exists", unknown)
    with pytest.raises(RuntimeError, match="purge is not supported"):
        ctx.execute("DROP TABLE IF EXISTS nope PURGE")


@pytest.mark.write_disk
@pytest.mark.parametrize("if_exists", ["", "IF EXISTS "])
def test_drop_catalog_table_purge_fails_after_drop(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch, if_exists: str
) -> None:
    error = PermissionError("manifest list not readable")

    def drop_then_fail(identifier: Any) -> None:
        lake.drop_table(identifier)
        raise error

    # a purge that drops the table, then fails deleting its files, is reported
    monkeypatch.setattr(lake, "purge_table", drop_then_fail)
    with pytest.raises(PermissionError) as exc:
        lake_ctx(lake).execute(f"DROP TABLE {if_exists}orders PURGE")
    assert exc.value is error
    assert exc.value.__notes__ == [
        "dropping table 'orders' failed: catalog table 'tpch.orders' of catalog 'lake'"
    ]
    assert not lake.table_exists(("tpch", "orders"))


@pytest.mark.write_disk
def test_drop_catalog_table_checks(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NoSuchTableError

    checks: list[Any] = []

    def unknown(identifier: Any) -> bool:
        checks.append(identifier)
        msg = "not allowed"
        raise PermissionError(msg)

    def deny(identifier: Any) -> None:
        msg = "drop not allowed"
        raise PermissionError(msg)

    ctx = lake_ctx(lake)
    monkeypatch.setattr(lake, "table_exists", unknown)
    # an existence check that fails lets the purge run
    ctx.execute("DROP TABLE orders PURGE")
    assert checks == [("tpch", "orders")]
    # a plain drop is not checked first, and its failure keeps its own error
    monkeypatch.setattr(lake, "drop_table", deny)
    with pytest.raises(PermissionError, match="drop not allowed") as exc:
        ctx.execute("DROP TABLE IF EXISTS lineitem")
    assert exc.value.__notes__ == [
        "dropping table 'lineitem' failed: catalog table 'tpch.lineitem' of "
        "catalog 'lake'"
    ]
    assert checks == [("tpch", "orders")]

    # a table dropped by another writer after the check
    def gone(identifier: Any) -> None:
        raise NoSuchTableError(str(identifier))

    monkeypatch.setattr(lake, "table_exists", lambda identifier: True)
    monkeypatch.setattr(lake, "purge_table", gone)
    ctx.execute("DROP TABLE IF EXISTS lineitem PURGE")
    with pytest.raises(NoSuchTableError, match="table 'lineitem' does not exist"):
        ctx.execute("DROP TABLE lineitem PURGE")


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
@pytest.mark.parametrize(
    ("name", "namespace"),
    [
        ("lake.sales", ("sales",)),
        ("LAKE.sales", ("sales",)),
        # of the default catalog, and not under its default namespace `tpch`
        ("sales", ("sales",)),
        ('lake."Sales"', ("Sales",)),
        ("lake.tpch.sales", ("tpch", "sales")),
        ("prod.sales", ("prod", "sales")),
    ],
)
def test_create_catalog_schema(
    lake: CountingCatalog, name: str, namespace: tuple[str, ...]
) -> None:
    ctx = lake_ctx(lake)
    res = ctx.execute(f"CREATE SCHEMA {name}", eager=True)
    full_name = ".".join(("lake", *namespace))
    assert res.item() == f"CREATE SCHEMA {full_name}"
    assert lake.namespace_exists(namespace)
    ctx.execute(f"CREATE TABLE {name}.t (x INT)")
    assert lake.table_exists((*namespace, "t"))


@pytest.mark.write_disk
def test_create_catalog_schema_properties(
    lake: CountingCatalog, tmp_path: Path
) -> None:
    ctx = lake_ctx(lake)
    location = f"file://{tmp_path}/sales_here"
    ctx.execute(
        f"CREATE SCHEMA sales WITH ('location' = '{location}', 'owner' = 'me', 'n' = 1)"
    )
    properties = lake.load_namespace_properties(("sales",))
    assert {"location": location, "owner": "me", "n": "1"}.items() <= properties.items()
    # tables of the namespace go under its location
    ctx.execute("CREATE TABLE sales.t (x INT)")
    assert lake.load_table(("sales", "t")).location().startswith(location)


@pytest.mark.write_disk
def test_create_existing_catalog_schema(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NamespaceAlreadyExistsError

    ctx = lake_ctx(lake)
    with pytest.raises(
        NamespaceAlreadyExistsError,
        match="schema 'tpch' already exists: namespace 'tpch' of catalog 'lake'",
    ):
        ctx.execute("CREATE SCHEMA tpch")
    # an existing namespace is left as it is
    res = ctx.execute("CREATE SCHEMA IF NOT EXISTS tpch WITH ('k' = 'v')", eager=True)
    assert res.item() == "CREATE SCHEMA lake.tpch"
    assert "k" not in lake.load_namespace_properties(("tpch",))
    assert lake.table_exists(("tpch", "lineitem"))

    # created meanwhile, by another writer
    def created_meanwhile(*args: Any, **kwargs: Any) -> None:
        msg = "created meanwhile"
        raise NamespaceAlreadyExistsError(msg)

    monkeypatch.setattr(lake, "namespace_exists", lambda namespace: False)
    monkeypatch.setattr(lake, "create_namespace", created_meanwhile)
    ctx.execute("CREATE SCHEMA IF NOT EXISTS tpch")
    with pytest.raises(NamespaceAlreadyExistsError, match="already exists"):
        ctx.execute("CREATE SCHEMA tpch")


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("name", "namespace"),
    [
        ("lake.empty", ("empty",)),
        ("empty", ("empty",)),
        ("LAKE.empty", ("empty",)),
        ("lake.prod.tpch.empty", ("prod", "tpch", "empty")),
    ],
)
def test_drop_catalog_schema(
    lake: CountingCatalog, name: str, namespace: tuple[str, ...]
) -> None:
    from pyiceberg.exceptions import NoSuchNamespaceError

    lake.create_namespace(namespace)
    ctx = lake_ctx(lake)
    res = ctx.execute(f"DROP SCHEMA {name} RESTRICT", eager=True)
    assert res.item() == f"DROP SCHEMA lake.{'.'.join(namespace)}"
    assert not lake.namespace_exists(namespace)
    assert lake.table_exists(("prod", "tpch", "lineitem"))

    res = ctx.execute(f"DROP SCHEMA IF EXISTS {name}", eager=True)
    assert res.item() == f"DROP SCHEMA lake.{'.'.join(namespace)}"
    with pytest.raises(
        NoSuchNamespaceError,
        match=f"schema '{name}' does not exist: .* identifiers are case-sensitive",
    ):
        ctx.execute(f"DROP SCHEMA {name}")


@pytest.mark.write_disk
@pytest.mark.parametrize("if_exists", ["", "IF EXISTS "])
def test_drop_catalog_schema_not_empty(lake: CountingCatalog, if_exists: str) -> None:
    from pyiceberg.exceptions import NamespaceNotEmptyError

    ctx = lake_ctx(lake)
    with pytest.raises(
        NamespaceNotEmptyError,
        match=r"schema 'tpch' is not empty: it has tables 'tpch\.lineitem', "
        r"'tpch\.orders'; drop them first",
    ):
        ctx.execute(f"DROP SCHEMA {if_exists}tpch")
    # a catalog may drop a namespace that has child namespaces, and change nothing
    with pytest.raises(
        NamespaceNotEmptyError,
        match=r"schema 'prod' is not empty: it has child schemas 'prod\.tpch';",
    ):
        ctx.execute(f"DROP SCHEMA {if_exists}prod")
    assert lake.table_exists(("tpch", "lineitem"))
    assert lake.table_exists(("prod", "tpch", "lineitem"))


@pytest.mark.write_disk
def test_drop_catalog_schema_tables_and_children(lake: CountingCatalog) -> None:
    from pyiceberg.exceptions import NamespaceNotEmptyError

    ctx = lake_ctx(lake)
    for name in ["t1", "t2", "t3", "t4"]:
        ctx.execute(f"CREATE TABLE prod.{name} (id BIGINT)")
    with pytest.raises(
        NamespaceNotEmptyError,
        match=r"it has tables 'prod\.t1', 'prod\.t2', 'prod\.t3', \.\.\. and child "
        r"schemas 'prod\.tpch';",
    ):
        ctx.execute("DROP SCHEMA prod")
    assert lake.namespace_exists(("prod",))


@pytest.mark.write_disk
def test_drop_catalog_schema_cascading_catalog(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NamespaceNotEmptyError

    drops: list[Any] = []
    # some catalogs drop a namespace together with its tables
    monkeypatch.setattr(lake, "drop_namespace", drops.append)
    with pytest.raises(NamespaceNotEmptyError, match="schema 'tpch' is not empty"):
        lake_ctx(lake).execute("DROP SCHEMA tpch")
    assert drops == []


@pytest.mark.write_disk
def test_drop_catalog_schema_filled_meanwhile(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NamespaceNotEmptyError

    def refuse(namespace: Any) -> None:
        msg = "namespace is not empty"
        raise NamespaceNotEmptyError(msg)

    # a catalog that refuses a table created between the listing and the drop
    monkeypatch.setattr(lake, "list_tables", lambda namespace: [])
    monkeypatch.setattr(lake, "drop_namespace", refuse)
    with pytest.raises(
        NamespaceNotEmptyError,
        match="schema 'tpch' is not empty: drop its tables and child schemas first",
    ):
        lake_ctx(lake).execute("DROP SCHEMA tpch")
    assert lake.table_exists(("tpch", "lineitem"))


@pytest.mark.write_disk
def test_drop_catalog_schema_listing_fails(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        msg = "listing not allowed"
        raise RuntimeError(msg)

    # a failed listing of child namespaces is left to the drop
    lake.create_namespace(("empty",))
    monkeypatch.setattr(lake, "list_namespaces", fail)
    ctx = lake_ctx(lake)
    ctx.execute("DROP SCHEMA empty")
    assert not lake.namespace_exists(("empty",))


@pytest.mark.write_disk
@pytest.mark.parametrize("if_exists", ["", "IF EXISTS "])
def test_drop_catalog_schema_table_listing_fails(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch, if_exists: str
) -> None:
    from pyiceberg.exceptions import NoSuchNamespaceError

    def fail(*args: Any, **kwargs: Any) -> None:
        msg = "listing not allowed"
        raise RuntimeError(msg)

    def missing(*args: Any, **kwargs: Any) -> None:
        msg = "no such namespace"
        raise NoSuchNamespaceError(msg)

    lake.create_namespace(("empty",))
    ctx = lake_ctx(lake)
    for listing, error in [(fail, RuntimeError), (missing, NoSuchNamespaceError)]:
        monkeypatch.setattr(lake, "list_tables", listing)
        with pytest.raises(error) as exc:
            ctx.execute(f"DROP SCHEMA {if_exists}empty")
        assert exc.value.__notes__ == [
            "schema 'empty' was not dropped: its tables could not be listed"
        ]
        assert lake.namespace_exists(("empty",))


@pytest.mark.write_disk
def test_drop_catalog_schema_listing_says_missing(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NoSuchNamespaceError

    def missing(*args: Any, **kwargs: Any) -> None:
        msg = "Namespace with name 'x%20y' does not exist"
        raise NoSuchNamespaceError(msg)

    # PyIceberg's REST client lists children of `x y` as `x%20y`, which is missing
    lake.create_namespace(("x y",))
    monkeypatch.setattr(lake, "list_namespaces", missing)
    ctx = lake_ctx(lake)
    ctx.execute('DROP SCHEMA "x y"')
    assert not lake.namespace_exists(("x y",))
    with pytest.raises(NoSuchNamespaceError, match="schema 'x y' does not exist"):
        ctx.execute('DROP SCHEMA "x y"')


@pytest.mark.write_disk
def test_drop_catalog_schema_beside_case_variant(lake: CountingCatalog) -> None:
    # the SQL catalog's listing compares case (its existence check does not)
    ctx = lake_ctx(lake)
    ctx.execute("CREATE SCHEMA sales")
    ctx.execute('CREATE SCHEMA "Sales".child')
    ctx.execute("DROP SCHEMA sales")
    assert lake.namespace_exists(("Sales", "child"))
    assert ("sales",) not in lake.list_namespaces()


@pytest.mark.write_disk
@pytest.mark.parametrize("quoted", ['"Sales"', '"Sales".sub'])
def test_drop_catalog_schema_quoted(lake: CountingCatalog, quoted: str) -> None:
    namespace = tuple(part.strip('"') for part in quoted.split("."))
    lake.create_namespace(namespace)
    lake_ctx(lake).execute(f"DROP SCHEMA lake.{quoted}")
    assert not lake.namespace_exists(namespace)


@pytest.mark.write_disk
def test_create_catalog_schema_without_parent(lake: CountingCatalog) -> None:
    # passed to the catalog: the SQL catalog creates it, REST catalogs refuse
    lake_ctx(lake).execute("CREATE SCHEMA lake.new.sub WITH ()")
    assert lake.namespace_exists(("new", "sub"))
    assert lake.namespace_exists(("new",))


@pytest.mark.write_disk
def test_catalog_schema_catalog_errors(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NoSuchNamespaceError

    def fail(*args: Any, **kwargs: Any) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    ctx = lake_ctx(lake)
    monkeypatch.setattr(lake, "create_namespace", fail)
    with pytest.raises(RuntimeError, match="boom") as exc:
        ctx.execute("CREATE SCHEMA sales")
    assert exc.value.__notes__ == [
        "schema 'sales' could not be created: namespace 'sales' of catalog 'lake'"
    ]
    monkeypatch.undo()

    lake.create_namespace(("empty",))
    monkeypatch.setattr(lake, "drop_namespace", fail)
    with pytest.raises(RuntimeError, match="boom") as exc:
        ctx.execute("DROP SCHEMA empty")
    assert exc.value.__notes__ == [
        "schema 'empty' could not be dropped: namespace 'empty' of catalog 'lake'"
    ]
    # a catalog that reports another error for a missing namespace
    monkeypatch.setattr(lake, "list_tables", lambda namespace: [])
    monkeypatch.setattr(lake, "list_namespaces", lambda namespace: [])
    ctx.execute("DROP SCHEMA IF EXISTS missing")
    with pytest.raises(NoSuchNamespaceError, match="schema 'missing' does not exist"):
        ctx.execute("DROP SCHEMA missing")


@pytest.mark.write_disk
def test_create_catalog_schema_unknown_existence(
    lake: CountingCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.exceptions import NamespaceAlreadyExistsError

    def unsupported(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    # a catalog that can't check existence answers the create
    ctx = lake_ctx(lake)
    monkeypatch.setattr(lake, "namespace_exists", unsupported)
    monkeypatch.setattr(
        lake,
        "create_namespace",
        lambda namespace, properties: created.append(namespace),
    )
    created: list[Any] = []
    ctx.execute("CREATE SCHEMA IF NOT EXISTS sales")
    assert created == [("sales",)]

    def exists(*args: Any, **kwargs: Any) -> None:
        msg = "exists"
        raise NamespaceAlreadyExistsError(msg)

    monkeypatch.setattr(lake, "create_namespace", exists)
    ctx.execute("CREATE SCHEMA IF NOT EXISTS sales")


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("statement", "error"),
    [
        ("CREATE SCHEMA lake", "'lake' is an attached catalog; write 'lake.<schema>'"),
        ("CREATE SCHEMA LAKE", "'LAKE' is an attached catalog; write 'lake.<schema>'"),
        ("DROP SCHEMA lake", "'lake' is an attached catalog"),
        ('CREATE SCHEMA lake.""', "has an empty part"),
        (
            "CREATE OR REPLACE SCHEMA lake.x",
            "`CREATE OR REPLACE SCHEMA` is not supported",
        ),
        ("CREATE SCHEMA lake.x AUTHORIZATION bob", "AUTHORIZATION` is not supported"),
        ("CREATE SCHEMA AUTHORIZATION bob", "AUTHORIZATION` is not supported"),
        ("CREATE SCHEMA lake.x OPTIONS (k = 1)", r"use `WITH \(\.\.\.\)` instead"),
        ("CREATE SCHEMA lake.x DEFAULT COLLATE 'und'", "`DEFAULT COLLATE`"),
        ("CREATE SCHEMA lake.x CLONE lake.y", "CLONE` is not supported"),
        (
            "CREATE SCHEMA lake.x WITH ('Location' = '/x')",
            "'Location' is not a schema option; did you mean 'location'",
        ),
        (
            "CREATE SCHEMA lake.x WITH ('location' = 'x')",
            "location must be an absolute path or a URI",
        ),
        ("CREATE SCHEMA lake.x WITH ('k' = 1, 'k' = 2)", "duplicate schema option 'k'"),
        (
            "CREATE SCHEMA lake.x WITH ('k' = 1 + 1)",
            "schema option values must be string, number or boolean literals",
        ),
        (
            "DROP SCHEMA tpch CASCADE",
            "CASCADE` is not supported; drop its tables first",
        ),
        ("DROP SCHEMA tpch PURGE", "PURGE` is not supported"),
        ("DROP SCHEMA tpch, prod", "DROP SCHEMA takes one schema name"),
    ],
)
def test_catalog_schema_errors(
    lake: CountingCatalog, statement: str, error: str
) -> None:
    namespaces = lake.list_namespaces()
    with pytest.raises(SQLInterfaceError, match=error):
        lake_ctx(lake).execute(statement)
    assert lake.list_namespaces() == namespaces
    assert lake.table_exists(("tpch", "lineitem"))


@pytest.mark.write_disk
def test_catalog_schema_without_catalog(lake: CountingCatalog) -> None:
    for statement in ("CREATE SCHEMA sales", "DROP SCHEMA sales"):
        kind = statement.rsplit(" ", 1)[0]
        with pytest.raises(
            SQLInterfaceError, match=f"`{kind}` is supported only for attached catalogs"
        ):
            pl.SQLContext().execute(statement)
        with pytest.raises(
            SQLInterfaceError,
            match=r"schema 'sales' has no catalog: no default catalog is set; "
            r"write '<catalog>\.sales'",
        ):
            lake_ctx(lake, default=False).execute(statement)
    assert not lake.namespace_exists(("sales",))


@pytest.mark.write_disk
def test_create_catalog_schema_of_default_namespace(lake: CountingCatalog) -> None:
    # the namespace of one-part table names can be created after registration
    ctx = lake_ctx(lake, default_namespace="sales")
    ctx.execute("CREATE SCHEMA sales")
    ctx.execute("CREATE TABLE t AS SELECT 1 AS x")
    assert ctx.execute("SELECT * FROM lake.sales.t", eager=True).item() == 1


def iceberg_columns(
    catalog: CountingCatalog, *identifier: str
) -> list[tuple[str, str]]:
    # nested field ids depend on the order of assignment
    schema = catalog.load_table(identifier).schema()
    return [(f.name, re.sub(r"\d+: ", "", str(f.field_type))) for f in schema.fields]


@pytest.mark.write_disk
@pytest.mark.parametrize("name", ["lake.tpch.lineitem", "tpch.lineitem", "lineitem"])
def test_alter_catalog_table_columns(lake: CountingCatalog, name: str) -> None:
    ctx = lake_ctx(lake)
    res = ctx.execute(f"ALTER TABLE {name} ADD COLUMN w BIGINT", eager=True)
    assert res.item() == "ALTER TABLE lake.tpch.lineitem"
    assert iceberg_columns(lake, "tpch", "lineitem") == [
        ("k", "long"),
        ("v", "string"),
        ("w", "long"),
    ]
    # a later statement reads the new schema
    expected = LINEITEM.with_columns(w=pl.lit(None, pl.Int64))
    assert_frame_equal(
        ctx.execute(f"SELECT * FROM {name}", eager=True),
        expected,
        check_row_order=False,
    )

    ctx.execute(f"ALTER TABLE {name} RENAME COLUMN v TO v2")
    ctx.execute(f"ALTER TABLE {name} DROP COLUMN w")
    assert_frame_equal(
        ctx.execute(f"SELECT * FROM {name}", eager=True),
        LINEITEM.rename({"v": "v2"}),
        check_row_order=False,
    )


@pytest.mark.write_disk
def test_alter_catalog_table_column_type(lake: CountingCatalog) -> None:
    from pyiceberg.schema import Schema
    from pyiceberg.types import FloatType, IntegerType, NestedField

    schema = Schema(
        NestedField(1, "c", IntegerType()), NestedField(2, "f", FloatType())
    )
    table = lake.create_table(("tpch", "promote"), schema=schema)
    pl.DataFrame(
        {"c": [7], "f": [1.5]}, schema={"c": pl.Int32, "f": pl.Float32}
    ).write_iceberg(table, mode="append")
    ctx = lake_ctx(lake)
    ctx.execute("ALTER TABLE promote ALTER COLUMN c SET DATA TYPE BIGINT")
    ctx.execute("ALTER TABLE promote ALTER COLUMN f TYPE DOUBLE")
    # the type it has already: nothing to change
    ctx.execute("ALTER TABLE promote ALTER COLUMN c SET DATA TYPE BIGINT")
    assert iceberg_columns(lake, "tpch", "promote") == [("c", "long"), ("f", "double")]
    assert ctx.execute("SELECT * FROM promote", eager=True).rows() == [(7, 1.5)]


@pytest.mark.write_disk
def test_alter_catalog_table_decimal_type(lake: CountingCatalog) -> None:
    from decimal import Decimal

    from polars.exceptions import InvalidOperationError

    ctx = lake_ctx(lake)
    ctx.execute("CREATE TABLE dec AS SELECT CAST(1.25 AS DECIMAL(10, 2)) AS d")
    ctx.execute("ALTER TABLE dec ALTER COLUMN d SET DATA TYPE DECIMAL(12, 2)")
    # only the precision widens: the stored values have the old scale
    for target in ["DECIMAL(12, 3)", "DECIMAL(12, 1)"]:
        with pytest.raises(InvalidOperationError, match="Iceberg only widens"):
            ctx.execute(f"ALTER TABLE dec ALTER COLUMN d SET DATA TYPE {target}")
    assert iceberg_columns(lake, "tpch", "dec") == [("d", "decimal(12, 2)")]
    assert ctx.execute("SELECT d FROM dec", eager=True).item() == Decimal("1.25")


@pytest.mark.write_disk
def test_alter_catalog_table_properties(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.execute(
        "ALTER TABLE lineitem SET TBLPROPERTIES ('probed-at' = 'x', 'n' = 1, 'b' = true)"
    )
    properties = lake.load_table(("tpch", "lineitem")).properties
    assert {"probed-at": "x", "n": "1", "b": "true"}.items() <= properties.items()


@pytest.mark.write_disk
def test_alter_catalog_table_if_exists(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.execute("ALTER TABLE lineitem ADD COLUMN IF NOT EXISTS v BIGINT")
    ctx.execute("ALTER TABLE lineitem DROP COLUMN IF EXISTS nope")
    assert iceberg_columns(lake, "tpch", "lineitem") == [("k", "long"), ("v", "string")]
    res = ctx.execute("ALTER TABLE IF EXISTS nope ADD COLUMN w BIGINT", eager=True)
    assert res.item() == "ALTER TABLE lake.tpch.nope"
    with pytest.raises(NoSuchTableError, match="table 'nope' does not exist"):
        ctx.execute("ALTER TABLE nope ADD COLUMN w BIGINT")
    with pytest.raises(NoSuchTableError, match="identifiers are case-sensitive"):
        ctx.execute("ALTER TABLE LINEITEM ADD COLUMN w BIGINT")


@pytest.mark.write_disk
def test_alter_catalog_table_column_errors(lake: CountingCatalog) -> None:
    from polars.exceptions import ColumnNotFoundError, DuplicateError

    ctx = lake_ctx(lake)
    ctx.execute("CREATE TABLE nested (s STRUCT<x INT>)")
    for statement, error, match in [
        ("lineitem ADD COLUMN v BIGINT", DuplicateError, "column 'v' already exists"),
        ("lineitem RENAME COLUMN v TO k", DuplicateError, "column 'k' already exists"),
        ("lineitem RENAME COLUMN v TO v", DuplicateError, "column 'v' already exists"),
        (
            "lineitem DROP COLUMN V",
            ColumnNotFoundError,
            "column names are case-sensitive",
        ),
        ("lineitem RENAME COLUMN nope TO x", ColumnNotFoundError, "column 'nope'"),
        ("lineitem ALTER COLUMN nope TYPE BIGINT", ColumnNotFoundError, "'nope'"),
        ("lineitem ALTER COLUMN nope DROP NOT NULL", ColumnNotFoundError, "'nope'"),
        # a dotted name is a top-level column, not the nested field `s.x`
        ('nested DROP COLUMN "s.x"', ColumnNotFoundError, "column 's.x' not found"),
    ]:
        with pytest.raises(error, match=match):
            ctx.execute(f"ALTER TABLE {statement}")
    assert iceberg_columns(lake, "tpch", "lineitem") == [("k", "long"), ("v", "string")]
    assert iceberg_columns(lake, "tpch", "nested") == [("s", "struct<x: optional int>")]
    ctx.execute('ALTER TABLE nested ADD COLUMN "s.z" BIGINT')
    assert iceberg_columns(lake, "tpch", "nested") == [
        ("s", "struct<x: optional int>"),
        ("s.z", "long"),
    ]


@pytest.mark.write_disk
def test_alter_catalog_table_catalog_errors(lake: CountingCatalog) -> None:
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    from polars.exceptions import InvalidOperationError

    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "w", LongType()),
        identifier_field_ids=[1],
    )
    lake.create_table(("tpch", "keyed"), schema=schema)
    ctx = lake_ctx(lake)
    ctx.execute("CREATE TABLE parted (id BIGINT, p VARCHAR) PARTITION BY (p)")
    with pytest.raises(
        InvalidOperationError, match="'id' of table 'keyed' is an identifier"
    ):
        ctx.execute("ALTER TABLE keyed ALTER COLUMN id DROP NOT NULL")
    # what the catalog refuses passes through, with a note naming the statement
    for statement in ["parted DROP COLUMN p", "keyed DROP COLUMN id"]:
        table = statement.split()[0]
        with pytest.raises(Exception) as exc:
            ctx.execute(f"ALTER TABLE {statement}")
        assert exc.value.__notes__ == [
            f"table '{table}' could not be altered: catalog table 'tpch.{table}' of "
            "catalog 'lake'"
        ]
    assert iceberg_columns(lake, "tpch", "parted") == [("id", "long"), ("p", "string")]


@pytest.mark.write_disk
def test_alter_catalog_table_type_changes(lake: CountingCatalog) -> None:
    from polars.exceptions import InvalidOperationError

    ctx = lake_ctx(lake)
    ctx.execute("CREATE TABLE typed (s STRUCT<x INT>, l INT[], b VARBINARY, t VARCHAR)")
    # the type a column has already, nested ones included: nothing to change
    ctx.execute("ALTER TABLE typed ALTER COLUMN s TYPE STRUCT<x INT>")
    ctx.execute("ALTER TABLE typed ALTER COLUMN l TYPE INT[]")
    # what Iceberg doesn't allow, although PyIceberg takes some of it
    for column, target in [
        ("t", "VARBINARY"),
        ("b", "VARCHAR"),
        ("t", "BIGINT"),
        ("s", "STRUCT<x BIGINT>"),
        ("l", "BIGINT"),
    ]:
        with pytest.raises(InvalidOperationError, match="Iceberg only widens"):
            ctx.execute(f"ALTER TABLE typed ALTER COLUMN {column} TYPE {target}")
    with pytest.raises(InvalidOperationError, match="Iceberg only widens"):
        ctx.execute("ALTER TABLE lineitem ALTER COLUMN k SET DATA TYPE INT")
    assert iceberg_columns(lake, "tpch", "typed") == [
        ("s", "struct<x: optional int>"),
        ("l", "list<int>"),
        ("b", "binary"),
        ("t", "string"),
    ]


@pytest.mark.write_disk
def test_alter_catalog_table_invalid_decimal(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    for decimal in [
        "DECIMAL(2, 3)",
        "DECIMAL(10, 2147483648)",
        "STRUCT<x DECIMAL(2, 3)>",
    ]:
        with pytest.raises(TypeError, match="an Iceberg decimal has a precision"):
            ctx.execute(f"ALTER TABLE lineitem ADD COLUMN d {decimal}")
        with pytest.raises(TypeError, match="an Iceberg decimal has a precision"):
            ctx.execute(f"CREATE TABLE bad (d {decimal})")
    ctx.execute("CREATE TABLE dec (d DECIMAL(10, 2))")
    with pytest.raises(TypeError, match="an Iceberg decimal has a precision"):
        ctx.execute("ALTER TABLE dec ALTER COLUMN d SET DATA TYPE DECIMAL(12, 13)")
    assert iceberg_columns(lake, "tpch", "lineitem") == [("k", "long"), ("v", "string")]
    assert iceberg_columns(lake, "tpch", "dec") == [("d", "decimal(10, 2)")]
    assert not lake.table_exists(("tpch", "bad"))


@pytest.mark.write_disk
def test_alter_catalog_table_name_mapping(
    lake: CountingCatalog, tmp_path: Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from polars.exceptions import DuplicateError

    # a data file without field ids, read by name through the table's name mapping
    table = lake.create_table(
        ("tpch", "imported"),
        schema=pa.schema([("k", pa.int64()), ("cat", pa.string())]),
    )
    path = tmp_path / "imported.parquet"
    pq.write_table(pa.table({"k": [1], "cat": ["veg"]}), path)
    table.add_files([f"file://{path}"])
    ctx = lake_ctx(lake)
    ctx.execute("ALTER TABLE imported RENAME COLUMN cat TO label")
    # the old name still reads the file's column, so a new column must not take it
    with pytest.raises(DuplicateError, match="its name mapping reads older data"):
        ctx.execute("ALTER TABLE imported ADD COLUMN cat VARCHAR")
    with pytest.raises(DuplicateError, match="its name mapping reads older data"):
        ctx.execute("ALTER TABLE imported RENAME COLUMN k TO cat")
    # back to a name of the same field
    ctx.execute("ALTER TABLE imported RENAME COLUMN label TO cat")
    ctx.execute("ALTER TABLE imported RENAME COLUMN cat TO label")
    assert ctx.execute("SELECT * FROM imported", eager=True).rows() == [(1, "veg")]

    # a mapped name without a field id is in use too
    mapping = '[{"field-id": 1, "names": ["k"]}, {"names": ["ignored"]}]'
    lake.load_table(("tpch", "lineitem")).transaction().set_properties(
        {"schema.name-mapping.default": mapping}
    ).commit_transaction()
    with pytest.raises(DuplicateError, match="its name mapping reads older data"):
        ctx.execute("ALTER TABLE lineitem ADD COLUMN ignored BIGINT")
    with pytest.raises(DuplicateError, match="its name mapping reads older data"):
        ctx.execute("ALTER TABLE lineitem RENAME COLUMN v TO ignored")


@pytest.mark.write_disk
def test_alter_catalog_table_drop_only_column(lake: CountingCatalog) -> None:
    from polars.exceptions import InvalidOperationError

    ctx = lake_ctx(lake)
    ctx.execute("ALTER TABLE lineitem DROP COLUMN v")
    with pytest.raises(InvalidOperationError, match="'k' is the only column"):
        ctx.execute("ALTER TABLE lineitem DROP COLUMN k")
    assert iceberg_columns(lake, "tpch", "lineitem") == [("k", "long")]


@pytest.mark.write_disk
def test_alter_catalog_table_drop_not_null(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.execute("CREATE TABLE req (a BIGINT NOT NULL, b BIGINT DEFAULT NULL)")
    ctx.execute("ALTER TABLE req ALTER COLUMN a DROP NOT NULL")
    ctx.execute("ALTER TABLE req ADD COLUMN c BIGINT NULL")
    ctx.execute("ALTER TABLE req ADD COLUMN d VARCHAR DEFAULT NULL")
    fields = lake.load_table(("tpch", "req")).schema().fields
    assert [(field.name, field.required) for field in fields] == [
        ("a", False),
        ("b", False),
        ("c", False),
        ("d", False),
    ]


@pytest.mark.write_disk
def test_alter_catalog_table_added_types(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    for column in [
        "a SMALLINT",
        "b REAL",
        "c DECIMAL(10, 2)",
        "d TIMESTAMPTZ",
        "e STRUCT<x BIGINT, y VARCHAR>",
        "f BIGINT[]",
    ]:
        ctx.execute(f"ALTER TABLE lineitem ADD COLUMN {column}")
    assert iceberg_columns(lake, "tpch", "lineitem")[2:] == [
        ("a", "int"),
        ("b", "float"),
        ("c", "decimal(10, 2)"),
        ("d", "timestamptz"),
        ("e", "struct<x: optional long, y: optional string>"),
        ("f", "list<long>"),
    ]
    with pytest.raises(TypeError, match="ALTER TABLE"):
        ctx.execute("ALTER TABLE lineitem ADD COLUMN g UBIGINT")


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("statement", "error"),
    [
        ("ADD COLUMN w BIGINT NOT NULL", "`ADD COLUMN ... NOT NULL` is not supported"),
        (
            "ADD COLUMN w BIGINT DEFAULT 5",
            "takes only NULL or DEFAULT NULL; found `DEFAULT 5`",
        ),
        ("ADD COLUMN w BIGINT COMMENT 'x'", "found `COMMENT 'x'`"),
        (
            "ADD COLUMN w BIGINT FIRST",
            "`ADD COLUMN ... FIRST | AFTER` is not supported",
        ),
        (
            "ADD COLUMN w BIGINT AFTER k",
            "`ADD COLUMN ... FIRST | AFTER` is not supported",
        ),
        ("DROP COLUMN v CASCADE", "`DROP COLUMN ... CASCADE` is not supported"),
        ("ALTER COLUMN v SET NOT NULL", "`SET NOT NULL` is not supported"),
        (
            "ALTER COLUMN k SET DEFAULT 1",
            "`ALTER COLUMN k SET DEFAULT 1` is not supported",
        ),
        (
            "ADD COLUMN a BIGINT, ADD COLUMN b BIGINT",
            "takes one operation per statement",
        ),
        ("ADD COLUMN a BIGINT, DROP COLUMN v", "takes one operation per statement"),
        ("SET TBLPROPERTIES ('format-version' = '2')", "'format-version' is reserved"),
        ("SET TBLPROPERTIES ('uuid' = 'x')", "'uuid' is reserved"),
        ("SET TBLPROPERTIES ('location' = '/x')", "location is not a property"),
        ("SET TBLPROPERTIES ('Location' = '/x')", "did you mean 'location'"),
        ("SET TBLPROPERTIES ('a' = 1, 'a' = 2)", "duplicate table property 'a'"),
        ("RENAME TO t2", "`ALTER TABLE ... RENAME TO t2` is not supported"),
        ("ADD CONSTRAINT c PRIMARY KEY (k)", "is not supported for catalog tables"),
    ],
)
def test_alter_catalog_table_sql_errors(
    lake: CountingCatalog, statement: str, error: str
) -> None:
    with pytest.raises(SQLInterfaceError, match=error):
        lake_ctx(lake).execute(f"ALTER TABLE lineitem {statement}")
    assert iceberg_columns(lake, "tpch", "lineitem") == [("k", "long"), ("v", "string")]


def test_alter_table_without_catalog(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.register("frame", pl.LazyFrame({"a": [1]}))
    with pytest.raises(SQLInterfaceError, match="supported only for catalog tables"):
        ctx.execute("ALTER TABLE frame ADD COLUMN b BIGINT")
    # a registered table hides the catalog table of that name
    ctx.register("lineitem", pl.LazyFrame({"a": [1]}))
    with pytest.raises(
        SQLInterfaceError,
        match=r"'lineitem' is a registered table, write 'lake\.tpch\.lineitem'",
    ):
        ctx.execute("ALTER TABLE lineitem ADD COLUMN b BIGINT")
    with pytest.raises(SQLInterfaceError, match="no default catalog is set"):
        lake_ctx(lake, default=False).execute(
            "ALTER TABLE lineitem ADD COLUMN b BIGINT"
        )
    with pytest.raises(SQLInterfaceError, match="supported only for catalog tables"):
        pl.SQLContext().execute("ALTER TABLE t ADD COLUMN b BIGINT")
    with pytest.raises(SQLInterfaceError, match="`ALTER TABLE ONLY` is not supported"):
        lake_ctx(lake).execute("ALTER TABLE ONLY lineitem ADD COLUMN b BIGINT")


def table_rows(catalog: CountingCatalog, *identifier: str) -> list[tuple[Any, ...]]:
    rows = catalog.load_table(identifier).scan().to_arrow().to_pylist()
    # NULLs last within a column
    return sorted(
        (tuple(row.values()) for row in rows),
        key=lambda row: tuple((v is None, 0 if v is None else v) for v in row),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("name", ["lake.tpch.lineitem", "tpch.lineitem", "lineitem"])
def test_insert_catalog_table(lake: CountingCatalog, name: str) -> None:
    ctx = lake_ctx(lake)
    res = ctx.execute(f"INSERT INTO {name} VALUES (3, 'c'), (4, 'd')", eager=True)
    assert res.item() == "INSERT INTO lake.tpch.lineitem"
    assert table_rows(lake, "tpch", "lineitem") == [
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
    ]
    # the rows of a query, reading the table it writes (as it was before the write)
    ctx.execute(f"INSERT INTO {name} SELECT k + 10, v FROM {name} WHERE k < 3")
    ctx.execute(
        f"WITH s AS (SELECT 20 AS k, 'w' AS v) INSERT INTO {name} SELECT * FROM s"
    )
    ctx.execute(f"INSERT INTO TABLE {name} WITH s AS (SELECT 21, 'x') SELECT * FROM s")
    assert table_rows(lake, "tpch", "lineitem")[4:] == [
        (11, "a"),
        (12, "b"),
        (20, "w"),
        (21, "x"),
    ]
    # performed at execute, also for a lazy result
    ctx.execute(f"INSERT INTO {name} VALUES (30, 'z')", eager=False)
    assert ctx.execute(f"SELECT count(*) FROM {name}", eager=True).item() == 9


@pytest.mark.write_disk
def test_insert_catalog_table_columns(lake: CountingCatalog) -> None:
    from polars.exceptions import ColumnNotFoundError, DuplicateError

    ctx = lake_ctx(lake)
    ctx.register("src", pl.LazyFrame({"x": [5], "y": ["e"]}))
    ctx.execute("INSERT INTO lineitem (v, k) VALUES ('c', 3)")
    ctx.execute("INSERT INTO lineitem (k) SELECT x FROM src")
    ctx.execute("INSERT INTO lineitem SELECT * FROM src")
    assert table_rows(lake, "tpch", "lineitem")[2:] == [(3, "c"), (5, "e"), (5, None)]
    for statement, error, match in [
        ("(k, k) VALUES (1, 2)", DuplicateError, "'k' is listed more than once"),
        (
            "(k, K) VALUES (1, 2)",
            ColumnNotFoundError,
            "column names are case-sensitive",
        ),
        (
            "VALUES (1)",
            SQLInterfaceError,
            "writes 2 columns of table 'lineitem', but the query gives 1",
        ),
        ("(k) VALUES (1, 'a')", SQLInterfaceError, "writes 1 columns"),
        ("(lineitem.k) VALUES (1)", SQLInterfaceError, "must be unqualified"),
    ]:
        with pytest.raises(error, match=match):
            ctx.execute(f"INSERT INTO lineitem {statement}")
    assert len(table_rows(lake, "tpch", "lineitem")) == 5


@pytest.mark.write_disk
def test_insert_catalog_table_overwrite(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    res = ctx.execute("INSERT OVERWRITE TABLE lineitem VALUES (7, 'g')", eager=True)
    assert res.item() == "INSERT OVERWRITE lake.tpch.lineitem"
    assert table_rows(lake, "tpch", "lineitem") == [(7, "g")]
    # every partition, whichever ones the new rows are in
    ctx.execute("CREATE TABLE parted (k BIGINT, p VARCHAR) PARTITION BY (p)")
    ctx.execute("INSERT INTO parted VALUES (1, 'a'), (2, 'b')")
    ctx.execute("INSERT OVERWRITE parted SELECT k + 10, p FROM parted WHERE p = 'a'")
    assert table_rows(lake, "tpch", "parted") == [(11, "a")]
    ctx.execute("INSERT OVERWRITE parted SELECT * FROM parted WHERE false")
    assert table_rows(lake, "tpch", "parted") == []


@pytest.mark.write_disk
def test_insert_catalog_table_partitioned(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.execute(
        "CREATE TABLE parted (id BIGINT, v BIGINT, p VARCHAR) PARTITION BY (p, bucket(4, id))"
    )
    ctx.execute("INSERT INTO parted VALUES (1, 10, 'a'), (2, 20, 'a'), (3, 30, 'b')")
    table = lake.load_table(("tpch", "parted"))
    partitions = {task.file.partition[0] for task in table.scan().plan_files()}
    assert partitions == {"a", "b"}
    assert table_rows(lake, "tpch", "parted") == [
        (1, 10, "a"),
        (2, 20, "a"),
        (3, 30, "b"),
    ]


@pytest.mark.write_disk
def test_insert_catalog_table_casts(lake: CountingCatalog) -> None:
    from decimal import Decimal

    from polars.exceptions import InvalidOperationError, SchemaError

    ctx = lake_ctx(lake)
    ctx.execute(
        "CREATE TABLE typed (i INT, l BIGINT, x DOUBLE, d DECIMAL(10, 2), dt DATE, "
        "ts TIMESTAMPTZ, li BIGINT[])"
    )
    ctx.register("floats", pl.LazyFrame({"f": [0.5, 1.5, 2.5, -0.5, -1.5]}))
    # a fractional decimal rounds half away from zero, a float half to even
    ctx.execute(
        "INSERT INTO typed (i) VALUES (0.5), (1.5), (2.5), (-0.5), (-1.5), (1.7)"
    )
    ctx.execute("INSERT INTO typed (l) SELECT f FROM floats")
    ctx.execute(
        "INSERT INTO typed (x, d, dt, ts, li) SELECT 3, 1.239, "
        "CAST('2026-01-03 10:00:00' AS TIMESTAMP), CAST('2026-01-02 03:04:05' AS TIMESTAMP), "
        "[1, 2]"
    )
    df = ctx.execute("SELECT * FROM typed", eager=True)
    assert sorted(df["i"].drop_nulls().to_list()) == [-2, -1, 1, 2, 2, 3]
    assert sorted(df["l"].drop_nulls().to_list()) == [-2, 0, 0, 2, 2]
    row = df.filter(pl.col("x").is_not_null()).row(0, named=True)
    assert row["x"] == 3.0
    assert row["d"] == Decimal("1.24")
    assert row["dt"] == date(2026, 1, 3)
    assert str(row["ts"]) == "2026-01-02 03:04:05+00:00"
    assert row["li"] == [1, 2]
    rows = len(df)
    # other kinds of values need an explicit CAST
    for column, value in [
        ("i", "'8'"),
        ("dt", "'2026-01-03'"),
        ("x", "true"),
        ("li", "[1.5]"),
    ]:
        with pytest.raises(SchemaError, match=f"column '{column}' of table 'typed'"):
            ctx.execute(f"INSERT INTO typed ({column}) SELECT {value}")
    # a value that does not fit is reported by its column, and nothing is written
    for column, value in [("i", "2147483648"), ("d", "123456789.5")]:
        with pytest.raises(InvalidOperationError, match=f"in column '{column}'"):
            ctx.execute(f"INSERT INTO typed ({column}) VALUES ({value})")
    assert len(ctx.execute("SELECT * FROM typed", eager=True)) == rows


@pytest.mark.write_disk
def test_insert_catalog_table_conversions(lake: CountingCatalog) -> None:
    from datetime import datetime

    from polars.exceptions import InvalidOperationError, SchemaError

    ctx = lake_ctx(lake)
    ctx.execute(
        "CREATE TABLE conv (ts TIMESTAMP, tz TIMESTAMPTZ, l BIGINT[], "
        "s STRUCT<a BIGINT, b VARCHAR>)"
    )
    ctx.execute("CREATE TABLE ints (i INT, r REAL)")
    # pattern-like column names are names, matched by position
    ctx.register("odd", pl.LazyFrame({"*": [1], "^v$": ["x"]}))
    ctx.execute("INSERT INTO lineitem SELECT * FROM odd")
    assert (1, "x") in table_rows(lake, "tpch", "lineitem")
    # overflow is checked after rounding
    with pytest.raises(InvalidOperationError, match="in column 'i'"):
        ctx.execute("INSERT INTO ints (i) VALUES (2147483647.5)")
    # a decimal with no integer digits rounds too
    ctx.execute("INSERT INTO ints (i) SELECT CAST(0.5 AS DECIMAL(38, 38))")
    ctx.execute("INSERT INTO ints (i) SELECT CAST(9.5 AS DECIMAL(38, 37))")
    assert sorted(ctx.execute("SELECT i FROM ints", eager=True)["i"].to_list()) == [
        1,
        10,
    ]
    # a double into a float column could overflow unchecked: an explicit CAST
    ctx.register("big", pl.LazyFrame({"f": [1e100]}))
    with pytest.raises(SchemaError, match="column 'r' of table 'ints'"):
        ctx.execute("INSERT INTO ints (r) SELECT f FROM big")
    ctx.register("u128", pl.LazyFrame({"u": pl.Series([2**128 - 1], dtype=pl.UInt128)}))
    with pytest.raises(SchemaError, match="column 'r' of table 'ints'"):
        ctx.execute("INSERT INTO ints (r) SELECT u FROM u128")
    # an instant in another time zone keeps the instant; nested values widen
    local = datetime(2026, 1, 2, 3, 4, 5)
    src = pl.LazyFrame(
        {"ts": [local], "tz": [local], "l": [[1, 2]], "s": [{"a": 1, "b": None}]},
        schema={
            "ts": pl.Datetime("ns"),
            "tz": pl.Datetime("ns"),
            "l": pl.List(pl.Int32),
            "s": pl.Struct({"a": pl.Int32, "b": pl.Null}),
        },
    ).with_columns(pl.col("ts", "tz").dt.replace_time_zone("Europe/Amsterdam"))
    ctx.register("src", src)
    ctx.execute("INSERT INTO conv SELECT * FROM src")
    row = ctx.execute("SELECT * FROM conv", eager=True).row(0, named=True)
    assert row["tz"] == datetime(2026, 1, 2, 2, 4, 5, tzinfo=UTC)
    assert row["ts"] == datetime(2026, 1, 2, 2, 4, 5)
    assert row["l"] == [1, 2]
    assert row["s"] == {"a": 1, "b": None}


@pytest.mark.write_disk
def test_insert_catalog_table_required(lake: CountingCatalog) -> None:
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    from polars.exceptions import InvalidOperationError

    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "v", StringType()),
    )
    lake.create_table(("tpch", "req"), schema=schema)
    ctx = lake_ctx(lake)
    ctx.execute("INSERT INTO req (id) VALUES (1)")
    with pytest.raises(
        InvalidOperationError, match="'id' of table 'req' is NOT NULL and has no"
    ):
        ctx.execute("INSERT INTO req (v) VALUES ('a')")
    with pytest.raises(InvalidOperationError, match="'id' of table 'req' is NOT NULL"):
        ctx.execute("INSERT INTO req VALUES (NULL, 'a')")
    # a NULL from the query fails the write
    with pytest.raises(Exception, match="NULL") as exc:
        ctx.execute("INSERT INTO req SELECT CAST(NULL AS BIGINT), 'a'")
    assert "could not be written by INSERT INTO" in exc.value.__notes__[-1]
    assert table_rows(lake, "tpch", "req") == [(1, None)]


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("statement", "error"),
    [
        (
            "INSERT INTO lineitem VALUES (1, 'a') RETURNING k",
            "`RETURNING` is not supported",
        ),
        (
            "INSERT INTO lineitem VALUES (1, 'a') ON CONFLICT DO NOTHING",
            "`ON CONFLICT`",
        ),
        (
            "INSERT OR REPLACE INTO lineitem VALUES (1, 'a')",
            "`INSERT OR ...` is not supported",
        ),
        ("REPLACE INTO lineitem VALUES (1, 'a')", "`REPLACE INTO` is not supported"),
        (
            "INSERT INTO lineitem PARTITION (v) SELECT 1, 'a'",
            "PARTITION \\(...\\)` is not supported",
        ),
        (
            "INSERT OVERWRITE lineitem PARTITION (v = 'a') SELECT 1",
            "partition_overwrite",
        ),
        ("INSERT INTO lineitem TABLE orders", "write `SELECT \\* FROM <name>`"),
        (
            "INSERT INTO lineitem DEFAULT VALUES",
            "`INSERT ... DEFAULT VALUES` is not supported",
        ),
        (
            "WITH s AS (SELECT 1, 'a') INSERT INTO lineitem WITH t AS (SELECT 2, 'b') SELECT * FROM t",
            "takes one WITH",
        ),
        (
            "INSERT INTO lineitem WITH q AS (SELECT 5, 'e') INSERT INTO orders SELECT * FROM q",
            "must not change a table",
        ),
    ],
)
def test_insert_catalog_table_sql_errors(
    lake: CountingCatalog, statement: str, error: str
) -> None:
    with pytest.raises(SQLInterfaceError, match=error):
        lake_ctx(lake).execute(statement)
    assert table_rows(lake, "tpch", "lineitem") == [(1, "a"), (2, "b")]


def test_insert_without_catalog(lake: CountingCatalog) -> None:
    from pyiceberg.exceptions import NoSuchTableError

    ctx = lake_ctx(lake)
    ctx.register("frame", pl.LazyFrame({"a": [1]}))
    with pytest.raises(
        SQLInterfaceError, match="`INSERT` is supported only for catalog"
    ):
        ctx.execute("INSERT INTO frame VALUES (2)")
    ctx.register("lineitem", pl.LazyFrame({"a": [1]}))
    with pytest.raises(
        SQLInterfaceError,
        match=r"`INSERT OVERWRITE` is supported only .* write 'lake\.tpch\.lineitem'",
    ):
        ctx.execute("INSERT OVERWRITE lineitem VALUES (2)")
    with pytest.raises(SQLInterfaceError, match="supported only for catalog tables"):
        pl.SQLContext().execute("INSERT INTO t VALUES (1)")
    with pytest.raises(NoSuchTableError, match="identifiers are case-sensitive"):
        lake_ctx(lake).execute("INSERT INTO LINEITEM VALUES (1, 'a')")


class RacingCatalog(CountingCatalog):  # type: ignore[misc]
    """A SQL catalog in which another writer appends a row just before a commit."""

    race: tuple[str, ...] | None = None

    def commit_table(self, table: Any, requirements: Any, updates: Any) -> Any:
        if (race := self.race) is not None and table.name() == race:
            import pyarrow as pa

            self.race = None
            other = self.load_table(race)
            other.append(
                pa.table({"k": [99], "v": ["raced"]}, schema=other.schema().as_arrow())
            )
        return super().commit_table(table, requirements, updates)


@pytest.mark.write_disk
def test_insert_catalog_table_concurrent(tmp_path: Path) -> None:
    from pyiceberg.exceptions import ValidationException

    lake = RacingCatalog(
        "lake", uri=f"sqlite:///{tmp_path / 'c.db'}", warehouse=f"file://{tmp_path}"
    )
    lake.create_namespace(("tpch",))
    table = lake.create_table(("tpch", "lineitem"), schema=LINEITEM.to_arrow().schema)
    LINEITEM.write_iceberg(table, mode="append")
    ctx = lake_ctx(lake)
    # an append is retried on the new snapshot: both rows land
    lake.race = ("tpch", "lineitem")
    ctx.execute("INSERT INTO lineitem VALUES (5, 'e')")
    assert table_rows(lake, "tpch", "lineitem") == [
        (1, "a"),
        (2, "b"),
        (5, "e"),
        (99, "raced"),
    ]
    # an overwrite would delete the other writer's row: refused, and its files deleted
    files = set(Path(tmp_path).rglob("*.parquet"))
    lake.race = ("tpch", "lineitem")
    with pytest.raises(ValidationException) as exc:
        ctx.execute("INSERT OVERWRITE lineitem VALUES (7, 'g')")
    assert "could not be written by INSERT OVERWRITE" in exc.value.__notes__[-1]
    assert len(table_rows(lake, "tpch", "lineitem")) == 5
    written = set(Path(tmp_path).rglob("*.parquet")) - files
    assert len(written) == 1  # the other writer's file only


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "statement",
    [
        "TRUNCATE TABLE lake.tpch.lineitem",
        "TRUNCATE tpch.lineitem",
        "TRUNCATE TABLE IF EXISTS lineitem",
    ],
)
def test_truncate_catalog_table(
    lake: CountingCatalog, tmp_path: Path, statement: str
) -> None:
    from pyiceberg.table.snapshots import Operation

    ctx = lake_ctx(lake)
    files = set(tmp_path.rglob("*.parquet"))
    res = ctx.execute(statement, eager=True)
    assert res.item() == "TRUNCATE TABLE lake.tpch.lineitem"
    assert table_rows(lake, "tpch", "lineitem") == []
    snapshot = lake.load_table(("tpch", "lineitem")).current_snapshot()
    assert snapshot.summary.operation == Operation.DELETE
    assert set(tmp_path.rglob("*.parquet")) == files  # no data files written
    assert ctx.execute("SELECT COUNT(*) FROM lineitem", eager=True).item() == 0
    assert len(table_rows(lake, "tpch", "orders")) == 3


@pytest.mark.write_disk
def test_truncate_catalog_table_empty_or_partitioned(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    ctx.execute("CREATE TABLE p (k INT, g VARCHAR) PARTITION BY (g)")
    # an empty table commits nothing, and PyIceberg's warnings stay silent
    ctx.execute("TRUNCATE TABLE p")
    assert lake.load_table(("tpch", "p")).current_snapshot() is None
    ctx.execute("INSERT INTO p VALUES (1, 'a'), (2, 'b'), (3, 'a')")
    ctx.execute("TRUNCATE TABLE p")
    assert table_rows(lake, "tpch", "p") == []
    snapshot = lake.load_table(("tpch", "p")).current_snapshot()
    ctx.execute("TRUNCATE TABLE p")
    assert lake.load_table(("tpch", "p")).current_snapshot() == snapshot
    ctx.execute("INSERT INTO p VALUES (4, 'c')")
    assert table_rows(lake, "tpch", "p") == [(4, "c")]
    # merge-on-read deletes are moot when every data file is dropped
    ctx.execute(
        "ALTER TABLE p SET TBLPROPERTIES ('write.delete.mode' = 'merge-on-read')"
    )
    ctx.execute("TRUNCATE TABLE p")
    assert table_rows(lake, "tpch", "p") == []


@pytest.mark.write_disk
@pytest.mark.parametrize("with_counted_file", [False, True])
def test_truncate_catalog_table_without_record_count(
    lake: CountingCatalog, with_counted_file: bool
) -> None:
    from pyiceberg.manifest import DataFile, DataFileContent
    from pyiceberg.table.snapshots import Operation

    # a data file whose writer did not record its row count is read to drop its rows
    file = next(iter(lake.load_table(("tpch", "lineitem")).scan().plan_files())).file
    uncounted = f"{file.file_path.removesuffix('.parquet')}-uncounted.parquet"
    shutil.copy(
        file.file_path.removeprefix("file://"), uncounted.removeprefix("file://")
    )
    table = lake.create_table(("tpch", "legacy"), schema=LINEITEM.to_arrow().schema)
    with table.transaction() as tx, tx.update_snapshot().fast_append() as append:
        if with_counted_file:
            append.append_data_file(file)
        append.append_data_file(
            DataFile.from_args(
                content=DataFileContent.DATA,
                file_path=uncounted,
                file_format=file.file_format,
                partition=file.partition,
                record_count=-1,
                file_size_in_bytes=file.file_size_in_bytes,
            )
        )
    assert len(table_rows(lake, "tpch", "legacy")) == 2 * (1 + with_counted_file)
    lake_ctx(lake).execute("TRUNCATE TABLE legacy")
    assert table_rows(lake, "tpch", "legacy") == []
    snapshots = lake.load_table(("tpch", "legacy")).metadata.snapshots
    # one commit: the counted file dropped whole, the other one read
    assert [s.summary.operation for s in snapshots] == [
        Operation.APPEND,
        *([Operation.DELETE] if with_counted_file else []),
        Operation.OVERWRITE,
    ]


@pytest.mark.write_disk
def test_truncate_catalog_table_resolution(lake: CountingCatalog) -> None:
    ctx = lake_ctx(lake)
    # a registered table of the name comes first, as in a read
    ctx.register("lineitem", pl.LazyFrame({"x": [1]}))
    ctx.execute("TRUNCATE TABLE lineitem")
    assert ctx.execute("SELECT * FROM lineitem", eager=True).is_empty()
    assert len(table_rows(lake, "tpch", "lineitem")) == 2
    ctx.execute("TRUNCATE TABLE lake.tpch.lineitem")
    assert table_rows(lake, "tpch", "lineitem") == []
    # a missing table: a no-op with IF EXISTS
    res = ctx.execute("TRUNCATE TABLE IF EXISTS lake.tpch.missing", eager=True)
    assert res.item() == "TRUNCATE TABLE lake.tpch.missing"
    with pytest.raises(NoSuchTableError, match="could not be truncated; catalog"):
        ctx.execute("TRUNCATE TABLE lake.tpch.missing")
    with pytest.raises(NoSuchTableError, match="identifiers are case-sensitive"):
        ctx.execute("TRUNCATE TABLE ORDERS")
    assert len(table_rows(lake, "tpch", "orders")) == 3


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("statement", "error"),
    [
        ("TRUNCATE TABLE orders CASCADE", "CASCADE/RESTRICT"),
        ("TRUNCATE TABLE orders RESTART IDENTITY", "RESTART/CONTINUE IDENTITY"),
        ("TRUNCATE TABLE orders, lake.tpch.lineitem", "exactly one table name"),
        ("TRUNCATE TABLE orders PARTITION (k = 1)", "'partitions'"),
    ],
)
def test_truncate_catalog_table_sql_errors(
    lake: CountingCatalog, statement: str, error: str
) -> None:
    with pytest.raises(SQLInterfaceError, match=re.escape(error)):
        lake_ctx(lake).execute(statement)
    assert len(table_rows(lake, "tpch", "orders")) == 3


@pytest.mark.write_disk
def test_truncate_catalog_table_concurrent(tmp_path: Path) -> None:
    from pyiceberg.exceptions import ValidationException

    lake = RacingCatalog(
        "lake", uri=f"sqlite:///{tmp_path / 'c.db'}", warehouse=f"file://{tmp_path}"
    )
    lake.create_namespace(("tpch",))
    table = lake.create_table(("tpch", "lineitem"), schema=LINEITEM.to_arrow().schema)
    LINEITEM.write_iceberg(table, mode="append")
    # the delete would drop the other writer's row unseen: refused
    lake.race = ("tpch", "lineitem")
    with pytest.raises(ValidationException) as exc:
        lake_ctx(lake).execute("TRUNCATE TABLE lineitem")
    assert "could not be truncated" in exc.value.__notes__[-1]
    assert table_rows(lake, "tpch", "lineitem") == [(1, "a"), (2, "b"), (99, "raced")]
    # with snapshot isolation, as the table asks, the other writer's rows are kept
    with lake.load_table(("tpch", "lineitem")).transaction() as tx:
        tx.set_properties({"write.delete.isolation-level": "snapshot"})
    lake.race = ("tpch", "lineitem")
    lake_ctx(lake).execute("TRUNCATE TABLE lineitem")
    assert table_rows(lake, "tpch", "lineitem") == [(99, "raced")]


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
