from __future__ import annotations

import sys
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import polars as pl
from polars._utils.various import parse_version
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

bridge = pytest.importorskip("polars_database._bridge")


class _GoogleRows(list[tuple[Any, ...]]):
    num_dml_affected_rows = None

    def __init__(self, schema: list[Any], rows: list[tuple[Any, ...]]) -> None:
        super().__init__(rows)
        self.schema = schema
        self.total_rows = len(rows)

    def _should_use_bqstorage(self, *_args: Any, **_kwargs: Any) -> bool:
        return False


@pytest.fixture
def bigquery_client() -> Iterator[Any]:
    bigquery = pytest.importorskip("google.cloud.bigquery")
    auth = pytest.importorskip("google.auth.credentials")
    with closing(
        bigquery.Client(
            project="billing-project",
            location="US",
            credentials=auth.AnonymousCredentials(),
        )
    ) as client:
        yield client


@pytest.fixture
def bigquery_engine(bigquery_client: Any) -> Iterator[Any]:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlalchemy_bigquery")
    engine = sa.create_engine(
        "bigquery:///?user_supplied_client=true",
        connect_args={"client": bigquery_client, "prefer_bqstorage_client": False},
    )
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(params=["dbapi", "sqlalchemy"])
def bigquery_connection(
    request: pytest.FixtureRequest, bigquery_client: Any
) -> Iterator[tuple[str, Any]]:
    if request.param == "sqlalchemy":
        yield "sqlalchemy", request.getfixturevalue("bigquery_engine")
    else:
        dbapi = pytest.importorskip("google.cloud.bigquery.dbapi")
        with closing(
            dbapi.connect(client=bigquery_client, prefer_bqstorage_client=False)
        ) as conn:
            yield "dbapi", conn


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("operation", ["eq", "eq_missing", "is_in"])
def test_bigquery_percent_filter_reaches_driver_unchanged(
    bigquery_connection: tuple[str, Any],
    bound: bool,
    operation: str,
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, connection = bigquery_connection
    sqlglot = pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    rows = [("a%%b",), ("a%b",), ("plain",)]
    fields = [bigquery.SchemaField("label", "STRING")]
    sqls: list[tuple[str, set[str]]] = []

    def query(sql: str, **_kwargs: Any) -> _GoogleRows:
        if "WHERE 1 = 0" in sql:
            return _GoogleRows(fields, [])
        where = sqlglot.parse_one(sql, read="bigquery").args.get("where")
        literals = (
            {
                term.this
                for term in where.find_all(sqlglot.exp.Literal)
                if term.is_string
            }
            if where is not None
            else set()
        )
        sqls.append((sql, literals))
        return _GoogleRows(
            fields, [row for row in rows if not literals or row[0] in literals]
        )

    monkeypatch.setattr(bigquery_client, "query_and_wait", query)
    # In SQLAlchemy's binding path text() already escapes % for the driver.
    source = (
        "source%%marker" if bound and target == "sqlalchemy" else "source%%%%marker"
    )
    sql = f"SELECT label FROM fixture WHERE '{source}' = '{source}'"
    params = None
    if bound:
        placeholder = ":n" if target == "sqlalchemy" else "%(n)s"
        sql += f" AND 1 = {placeholder}"
        params = {"n": 1}
    scan = pl.scan_database(sql, connection, params=params, partition_on=None)
    materialized = scan.collect()
    for value in ["a%%b", "a%b", "plain"]:
        col = pl.col("label")
        predicate = {
            "eq": col == value,
            "eq_missing": col.eq_missing(value),
            "is_in": col.is_in([value, "other"]),
        }[operation]
        filtered = scan.filter(predicate)
        pushed = "%%" not in value and (not bound or "%" not in value)
        sqls.clear()
        assert_frame_equal(filtered.collect(), materialized.filter(predicate))
        submitted, literals = sqls[-1]
        assert "'source%%marker'" in submitted
        assert literals == (
            {value, "other"}
            if pushed and operation == "is_in"
            else {value}
            if pushed
            else set()
        )


@pytest.mark.parametrize(
    ("source_kind", "bound"),
    [("query", False), ("query", True), ("table", False), ("unwrapped", False)],
)
def test_bigquery_percent_column_names(
    source_kind: str,
    bound: bool,
    bigquery_connection: tuple[str, Any],
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, connection = bigquery_connection
    sqlglot = pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    data = {"a%%b": 11, "a%b": 22}
    sqls: list[Any] = []

    def query(sql: str, **_kwargs: Any) -> _GoogleRows:
        statement = sqlglot.parse_one(sql, read="bigquery")
        sqls.append(statement)
        columns = [expression.output_name for expression in statement.expressions]
        if columns == ["*"]:
            columns = list(data)
        fields = [bigquery.SchemaField(name, "INTEGER") for name in columns]
        rows = [] if "WHERE 1 = 0" in sql else [tuple(data[name] for name in columns)]
        return _GoogleRows(fields, rows)

    monkeypatch.setattr(bigquery_client, "query_and_wait", query)
    source = "fixture"
    params = None
    if source_kind == "query":
        # text() escapes the caller's SQL on SQLAlchemy's parameter-binding path.
        aliases = (
            list(data)
            if bound and target == "sqlalchemy"
            else [name.replace("%", "%%") for name in data]
        )
        source = f"SELECT 11 AS `{aliases[0]}`, 22 AS `{aliases[1]}`"
        if bound:
            placeholder = ":n" if target == "sqlalchemy" else "%(n)s"
            source += f" WHERE 1 = {placeholder}"
            params = {"n": 1}
    scan = pl.scan_database(
        source,
        connection,
        params=params,
        partition_on=None,
        pushdown=source_kind != "unwrapped",
        order_by=None if bound else "a%%b",
    )
    assert scan.collect().rows() == [(11, 22)]
    if not bound:
        assert sqls[-1].args["order"].expressions[0].this.name == "a%%b"
    for name, value in data.items():
        assert scan.select(name).collect().item() == value


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("target", ["uri", "engine", "connection"])
def test_bigquery_sqlalchemy_typed_results(
    target: str,
    empty: bool,
    bigquery_engine: Any,
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    QueryJobConfig, SchemaField = bigquery.QueryJobConfig, bigquery.SchemaField

    fields = [
        SchemaField("n", "NUMERIC"),
        SchemaField("ts", "TIMESTAMP"),
        SchemaField(
            "nested",
            "RECORD",
            mode="REPEATED",
            fields=(SchemaField("ts", "TIMESTAMP"),),
        ),
    ]
    exact = Decimal("12345678901234567890.123456789")
    timestamp = datetime(2025, 1, 1, 12, 34, 56, 123456, tzinfo=UTC)
    rows = [(exact, timestamp, [{"ts": timestamp}])]

    def query(sql: str, *, job_config: Any, **_kwargs: Any) -> _GoogleRows:
        if sql == "SELECT 1":
            return _GoogleRows([SchemaField("id", "INTEGER")], [(1,)])
        assert job_config.maximum_bytes_billed == 1024
        return _GoogleRows(fields, [] if empty or "WHERE 1 = 0" in sql else rows)

    monkeypatch.setattr(bigquery_client, "query_and_wait", query)
    with bigquery_engine.connect() as conn:
        options: dict[str, Any] = {}
        if target == "uri":
            connection = str(bigquery_engine.url)
            options = {
                "driver": "sqlalchemy",
                "connect_options": {
                    "connect_args": {
                        "client": bigquery_client,
                        "prefer_bqstorage_client": False,
                    }
                },
            }
        else:
            connection = bigquery_engine if target == "engine" else conn
        scan = pl.scan_database(
            "SELECT n, ts, nested FROM a_table",
            connection,
            partition_on=None,
            batch_size=1,
            execute_options={
                "execution_options": {
                    "job_config": QueryJobConfig(maximum_bytes_billed=1024),
                    "stream_results": True,
                    "yield_per": 1,
                }
            },
            **options,
        )
        utc = pl.Datetime("us", "UTC")
        expected = pl.Schema(
            {
                "n": pl.Decimal(38, 9),
                "ts": utc,
                "nested": pl.List(pl.Struct({"ts": utc})),
            }
        )
        assert scan.collect_schema() == expected
        df = scan.collect()
        assert df.schema == expected
        assert df.rows() == ([] if empty else rows)
        assert conn.exec_driver_sql("SELECT 1").scalar_one() == 1


@pytest.mark.parametrize(
    "target",
    ["live", "factory", "uri", "client_uri", "sa_engine", "sa_live", "sa_uri"],
)
@pytest.mark.parametrize("operation", ["collect_schema", "explain", "collect"])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_bigquery_destination_rejected_before_query(
    target: str,
    operation: str,
    explicit_config: bool,
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    dbapi = pytest.importorskip("google.cloud.bigquery.dbapi")

    def unexpected_query(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("scan issued a query with a destination table")

    for method in ("query", "query_and_wait"):
        monkeypatch.setattr(type(bigquery_client), method, unexpected_query)
    cfg = bigquery.QueryJobConfig(
        destination="billing-project.dataset.table", write_disposition="WRITE_TRUNCATE"
    )
    native_options: dict[str, Any] = {"prefer_bqstorage_client": False}
    options: dict[str, Any] = {}

    if explicit_config:
        options["execute_options"] = (
            {"execution_options": {"job_config": cfg}}
            if target.startswith("sa_")
            else {"job_config": cfg}
        )
    else:
        bigquery_client.default_query_job_config = cfg

    if operation == "collect":
        options["schema"] = {"id": pl.Int64}

    def factory() -> Any:
        return dbapi.connect(client=bigquery_client, **native_options)

    with ExitStack() as stack:
        connection: Any
        if target.startswith("sa_"):
            sa = pytest.importorskip("sqlalchemy")
            pytest.importorskip("sqlalchemy_bigquery")
            uri = "bigquery:///?user_supplied_client=true"
            connect_args = {"client": bigquery_client, **native_options}
            if target == "sa_uri":
                connection = uri
                options["driver"] = "sqlalchemy"
                options["connect_options"] = {"connect_args": connect_args}
            else:
                engine = sa.create_engine(uri, connect_args=connect_args)
                stack.callback(engine.dispose)
                connection = (
                    engine
                    if target == "sa_engine"
                    else stack.enter_context(engine.connect())
                )
        elif target == "live":
            connection = stack.enter_context(closing(factory()))
        elif target == "factory":
            connection = factory
        else:
            options["driver"] = "bigquery"
            if target == "client_uri" or explicit_config:
                connection = "bigquery://"
                options["connect_options"] = {
                    "client": bigquery_client,
                    **native_options,
                }
            else:
                connection = "bigquery://billing-project/dataset"
                options["connect_options"] = {
                    "default_query_job_config": cfg,
                    **native_options,
                }

        with pytest.raises(pl.exceptions.ComputeError, match="BigQuery destination"):
            getattr(
                pl.scan_database(
                    "SELECT 17 AS id", connection, partition_on=None, **options
                ),
                operation,
            )()


def test_bigquery_sqlalchemy_uri_rejects_named_destination(
    bigquery_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlglot")
    dialect = pytest.importorskip("sqlalchemy_bigquery.base")
    create_client = Mock(return_value=bigquery_client)
    monkeypatch.setattr(dialect._helpers, "create_bigquery_client", create_client)
    query = Mock(side_effect=AssertionError("scan issued a BigQuery query"))
    monkeypatch.setattr(bigquery_client, "query", query)
    monkeypatch.setattr(bigquery_client, "query_and_wait", query)
    scan = pl.scan_database(
        "SELECT 17 AS id",
        "bigquery://billing-project?destination=billing-project.dataset.target",
        driver="sqlalchemy",
        partition_on=None,
    )
    with pytest.raises(pl.exceptions.ComputeError, match="BigQuery destination"):
        scan.collect_schema()
    create_client.assert_not_called()
    query.assert_not_called()


def test_bigquery_sqlalchemy_pool_pre_ping(
    bigquery_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlalchemy_bigquery")
    pytest.importorskip("sqlglot")
    SchemaField = pytest.importorskip("google.cloud.bigquery").SchemaField

    uri = "bigquery:///?user_supplied_client=true"
    connect_args = {"client": bigquery_client, "prefer_bqstorage_client": False}
    engine = sa.create_engine(uri, pool_pre_ping=True, connect_args=connect_args)
    query = Mock(side_effect=AssertionError("scan issued a BigQuery query"))
    monkeypatch.setattr(bigquery_client, "query", query)
    monkeypatch.setattr(bigquery_client, "query_and_wait", query)

    try:
        for checkout in (engine, engine.connect):
            with pytest.raises(ValueError, match=r"BigQuery.*pool_pre_ping"):
                pl.scan_database("SELECT 17 AS id", checkout, partition_on=None)

        with pytest.raises(ValueError, match=r"BigQuery.*pool_pre_ping"):
            pl.scan_database("SELECT 17 AS id", lambda: engine, partition_on=None)

        uri_scan = pl.scan_database(
            "SELECT 17 AS id",
            uri,
            driver="sqlalchemy",
            connect_options={"pool_pre_ping": True, "connect_args": connect_args},
            partition_on=None,
        )
        with pytest.raises(
            pl.exceptions.ComputeError, match=r"BigQuery.*pool_pre_ping"
        ):
            uri_scan.collect_schema()
        query.assert_not_called()

        def result(sql: str, **_kwargs: Any) -> _GoogleRows:
            return _GoogleRows(
                [SchemaField("id", "INTEGER")], [] if "WHERE 1 = 0" in sql else [(17,)]
            )

        monkeypatch.setattr(bigquery_client, "query_and_wait", result)
        with engine.connect() as conn:
            scan = pl.scan_database("SELECT 17 AS id", conn, partition_on=None)
            assert scan.collect().rows() == [(17,)]
            assert conn.exec_driver_sql("SELECT 17").scalar_one() == 17
    finally:
        engine.dispose()


@pytest.mark.parametrize("wrapper", ["partial", "closure"])
def test_bigquery_wrapped_factory_rejects_pool_pre_ping_before_checkout(
    wrapper: str, bigquery_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlalchemy_bigquery")
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    engine = sa.create_engine(
        "bigquery:///?user_supplied_client=true",
        pool_pre_ping=True,
        connect_args={"client": bigquery_client, "prefer_bqstorage_client": False},
    )

    def checkout() -> Any:
        return engine.connect()

    try:
        with engine.connect() as conn:
            assert not conn.closed
        bigquery_client.default_query_job_config = bigquery.QueryJobConfig(
            destination="billing-project.data.sink", write_disposition="WRITE_TRUNCATE"
        )
        query = Mock(side_effect=AssertionError("pool ping issued a BigQuery query"))
        monkeypatch.setattr(bigquery_client, "query", query)
        monkeypatch.setattr(bigquery_client, "query_and_wait", query)

        factory = partial(engine.connect) if wrapper == "partial" else checkout
        with pytest.raises(ValueError, match=r"BigQuery.*pool_pre_ping"):
            pl.scan_database("SELECT 17 AS id", factory, partition_on=None)
        query.assert_not_called()
    finally:
        engine.dispose()


@pytest.mark.parametrize("target", ["engine", "factory", "live"])
def test_bigquery_static_pool_preserves_job_options_and_checkout(
    target: str, bigquery_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlalchemy_bigquery")
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")

    default = bigquery.QueryJobConfig(default_dataset="billing-project.dataset_a")
    engine_options = bigquery.QueryJobConfig(
        default_dataset="billing-project.dataset_b", maximum_bytes_billed=1024
    )
    scan_options = bigquery.QueryJobConfig(
        default_dataset="billing-project.dataset_c", maximum_bytes_billed=2048
    )
    bigquery_client.default_query_job_config = default
    engine = sa.create_engine(
        "bigquery:///?user_supplied_client=true",
        poolclass=sa.pool.StaticPool,
        connect_args={"client": bigquery_client, "prefer_bqstorage_client": False},
        execution_options={"job_config": engine_options},
    )
    resets = Mock()
    sa.event.listen(engine, "reset", resets)
    seen: list[tuple[str, str, int]] = []

    def result(sql: str, *, job_config: Any, **_kwargs: Any) -> _GoogleRows:
        if "marker_error" in sql:
            msg = "intercepted BigQuery query failure"
            raise ValueError(msg)
        if ":val" in sql:
            pytest.fail("SQLAlchemy bind marker reached BigQuery")
        dataset = str(job_config.default_dataset)
        seen.append((sql, dataset, job_config.maximum_bytes_billed))
        empty = "WHERE 1 = 0" in sql
        value = 17 if "orders" not in sql else job_config.query_parameters[0].value
        return _GoogleRows(
            [bigquery.SchemaField("id", "INTEGER")],
            [] if empty else [(value,), (value,)],
        )

    monkeypatch.setattr(bigquery_client, "query_and_wait", result)
    try:
        with engine.connect() as user:
            held = user.exec_driver_sql("SELECT 17 AS id")
            transaction = user.get_transaction()
            assert transaction is not None
            raw = user.connection.driver_connection
            rollback = Mock(wraps=engine.dialect.do_rollback)
            monkeypatch.setattr(engine.dialect, "do_rollback", rollback)
            source: Any = {"engine": engine, "factory": engine.connect, "live": user}[
                target
            ]
            for supplied in (None, scan_options):
                seen.clear()
                scan = pl.scan_database(
                    "SELECT :val AS id FROM orders",
                    source,
                    params={"val": 19},
                    execute_options=(
                        {"execution_options": {"job_config": supplied}}
                        if supplied is not None
                        else None
                    ),
                    batch_size=1,
                    partition_on=None,
                )
                assert scan.collect_schema() == pl.Schema({"id": pl.Int64})
                assert scan.head(1).collect().rows() == [(19,)]
                expected = engine_options if supplied is None else supplied
                assert {(dataset, cap) for _, dataset, cap in seen} == {
                    (str(expected.default_dataset), expected.maximum_bytes_billed)
                }
            with pytest.raises(pl.exceptions.ComputeError, match="intercepted"):
                pl.scan_database(
                    "SELECT 17 AS marker_error",
                    source,
                    schema={"marker_error": pl.Int64},
                    partition_on=None,
                ).collect()
            assert user.get_transaction() is transaction
            assert transaction.is_active
            assert user.connection.driver_connection is raw
            assert held.fetchall() == [(17,), (17,)]
            assert user.exec_driver_sql("SELECT 17 AS id").first() == (17,)
            resets.assert_not_called()
            rollback.assert_not_called()
    finally:
        engine.dispose()


@pytest.mark.parametrize("pool", ["SingletonThreadPool", "AssertionPool"])
def test_bigquery_shared_factory_cannot_discard_checkout_options(
    pool: str, bigquery_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlalchemy_bigquery")
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")

    engine = sa.create_engine(
        "bigquery:///?user_supplied_client=true",
        poolclass=getattr(sa.pool, pool),
        connect_args={"client": bigquery_client, "prefer_bqstorage_client": False},
    )
    config = bigquery.QueryJobConfig(default_dataset="billing-project.dataset_b")
    query = Mock()
    monkeypatch.setattr(bigquery_client, "query_and_wait", query)
    checkout = Mock()
    sa.event.listen(engine, "checkout", checkout)

    def factory() -> Any:
        return engine.connect().execution_options(job_config=config)

    try:
        with pytest.raises(TypeError, match="established SQLAlchemy Connection"):
            pl.scan_database("SELECT id FROM orders", factory, partition_on=None)
        checkout.assert_not_called()
        query.assert_not_called()

        def result(sql: str, *, job_config: Any, **_kwargs: Any) -> _GoogleRows:
            if "orders" in sql:
                assert job_config.default_dataset == config.default_dataset
            return _GoogleRows([bigquery.SchemaField("id", "INTEGER")], [(17,)])

        query.side_effect = result
        with engine.connect().execution_options(job_config=config) as conn:
            # A SQLAlchemy scan registers the class-wide psycopg2 listener.
            assert pl.scan_database(
                "SELECT id FROM orders", conn, partition_on=None
            ).collect().rows() == [(17,)]
        for source in (engine, engine.connect, engine.raw_connection):
            assert pl.scan_database(
                "SELECT 17 AS id", source, partition_on=None
            ).collect().rows() == [(17,)]

        def set_job_config(conn: Any) -> None:
            conn.execution_options(job_config=config)

        sa.event.listen(engine, "engine_connect", set_job_config)
        query.reset_mock()
        with pytest.raises(TypeError, match="established SQLAlchemy Connection"):
            pl.scan_database("SELECT id FROM orders", engine, partition_on=None)
        query.assert_not_called()
    finally:
        engine.dispose()


@pytest.mark.parametrize("bound", [False, True])
def test_bigquery_sqlalchemy_session_job_config(
    bound: bool,
    bigquery_engine: Any,
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    orm = pytest.importorskip("sqlalchemy.orm")
    pytest.importorskip("sqlglot")
    if parse_version(sa.__version__) < (2, 1):
        pytest.skip("Session-wide execution options require SQLAlchemy 2.1")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    QueryJobConfig, SchemaField = bigquery.QueryJobConfig, bigquery.SchemaField

    def scan(session: Any) -> pl.LazyFrame:
        return pl.scan_database(
            "SELECT :answer AS id" if bound else "SELECT 17 AS id, ':literal%' AS s",
            session,
            params={"answer": 17} if bound else None,
            schema={"id": pl.Int64} if bound else {"id": pl.Int64, "s": pl.String},
            partition_on=None,
        )

    query = Mock(side_effect=AssertionError("scan issued a BigQuery query"))
    monkeypatch.setattr(bigquery_client, "query", query)
    monkeypatch.setattr(bigquery_client, "query_and_wait", query)
    destination = QueryJobConfig(destination="billing-project.dataset.table")
    with orm.Session(
        bigquery_engine, execution_options={"job_config": destination}
    ) as s:
        with pytest.raises(pl.exceptions.ComputeError, match="BigQuery destination"):
            scan(s).collect()
        assert not s.in_transaction()
    query.assert_not_called()

    def harmless(sql: str, *, job_config: Any, **_kwargs: Any) -> _GoogleRows:
        assert job_config.maximum_bytes_billed == 1024
        fields = [SchemaField("id", "INTEGER")]
        if bound:
            return _GoogleRows(fields, [(17,)])
        assert "':literal%'" in sql
        return _GoogleRows([*fields, SchemaField("s", "STRING")], [(17, ":literal%")])

    monkeypatch.setattr(bigquery_client, "query_and_wait", harmless)
    safe = QueryJobConfig(maximum_bytes_billed=1024)
    with orm.Session(bigquery_engine, execution_options={"job_config": safe}) as s:
        assert scan(s).collect().rows() == ([(17,)] if bound else [(17, ":literal%")])


@pytest.mark.parametrize("pool", ["QueuePool", "StaticPool", "AssertionPool"])
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("target", ["live", "factory"])
def test_bigquery_rejects_orm_execution_listeners_before_query(
    pool: str,
    bound: bool,
    target: str,
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    orm = pytest.importorskip("sqlalchemy.orm")
    pytest.importorskip("sqlalchemy_bigquery")
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    engine = sa.create_engine(
        "bigquery:///?user_supplied_client=true",
        poolclass=getattr(sa.pool, pool),
        connect_args={"client": bigquery_client, "prefer_bqstorage_client": False},
    )
    config = bigquery.QueryJobConfig(
        destination="billing-project.dataset.table", write_disposition="WRITE_TRUNCATE"
    )
    query = Mock(side_effect=AssertionError("scan issued a BigQuery query"))
    monkeypatch.setattr(bigquery_client, "query", query)
    monkeypatch.setattr(bigquery_client, "query_and_wait", query)

    def inject_destination(state: Any) -> None:
        state.update_execution_options(job_config=config)

    class RoutingSession(orm.Session):  # type: ignore[misc,name-defined]
        pass

    sa.event.listen(RoutingSession, "do_orm_execute", inject_destination)
    try:
        with RoutingSession(engine) as session:
            source: Any = (
                session if target == "live" else lambda: RoutingSession(engine)
            )
            # Shared sources can be refused at construction, and others before execute.
            with pytest.raises(
                (TypeError, pl.exceptions.ComputeError),
                match=r"do_orm_execute|established SQLAlchemy Connection",
            ):
                pl.scan_database(
                    "SELECT :answer AS id" if bound else "SELECT 17 AS id",
                    source,
                    params={"answer": 17} if bound else None,
                    partition_on=None,
                ).collect()
            assert not session.in_transaction()
        query.assert_not_called()
    finally:
        engine.dispose()


@pytest.mark.parametrize("target", ["live", "engine", "factory"])
@pytest.mark.parametrize(
    "event", ["before_execute", "before_cursor_execute", "begin", "do_execute"]
)
def test_bigquery_rejects_execution_listeners_before_query(
    target: str,
    event: str,
    bigquery_engine: Any,
    bigquery_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlglot")
    bigquery = pytest.importorskip("google.cloud.bigquery")
    config = bigquery.QueryJobConfig(
        destination="billing-project.dataset.table", write_disposition="WRITE_TRUNCATE"
    )
    query = Mock(side_effect=AssertionError("scan issued a BigQuery query"))
    monkeypatch.setattr(bigquery_client, "query", query)
    monkeypatch.setattr(bigquery_client, "query_and_wait", query)

    def inject_destination(*args: Any) -> None:
        bigquery_client.default_query_job_config = config

    with bigquery_engine.connect() as conn:
        if event == "do_execute":
            event_target = bigquery_engine.dialect
        else:
            event_target = conn if target == "live" else bigquery_engine
        sa.event.listen(event_target, event, inject_destination)
        try:
            source: Any = {
                "live": conn,
                "engine": bigquery_engine,
                "factory": bigquery_engine.connect,
            }[target]
            with pytest.raises(
                pl.exceptions.ComputeError,
                match="BigQuery SQLAlchemy execution or begin listeners",
            ):
                pl.scan_database(
                    "SELECT :answer AS id",
                    source,
                    params={"answer": 17},
                    partition_on=None,
                ).collect()
            assert bigquery_client.default_query_job_config is None
        finally:
            sa.event.remove(event_target, event, inject_destination)
    query.assert_not_called()


def test_bigquery_connection_module() -> None:
    bigquery = type(
        "Connection", (), {"__module__": "google.cloud.bigquery.dbapi.connection"}
    )
    subclass = type("CustomConnection", (bigquery,), {"close": lambda self: None})
    assert bridge.connection_info(subclass(), "live")[1] == "google.cloud.bigquery"
    assert bridge.connection_info(subclass, "factory")[1] == "google.cloud.bigquery"
    for module in ("google.cloud.storage", "google.cloud.bigquery_storage_v1"):
        other = type("Connection", (), {"__module__": module})
        assert bridge.connection_info(other(), "live")[1] == "google"


def test_bigquery_python_uri_opens_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = Mock()
    client = Mock()
    monkeypatch.setitem(
        sys.modules, "google.cloud.bigquery.dbapi", SimpleNamespace(connect=connect)
    )
    monkeypatch.setitem(
        sys.modules,
        "google.cloud.bigquery",
        SimpleNamespace(Client=client, QueryJobConfig=SimpleNamespace),
    )
    credentials, storage = object(), object()
    config = SimpleNamespace(default_dataset=None, use_query_cache=False)
    conn, close = bridge._connect_bigquery(
        "bigquery+bigquery://billing-project/Some_Dataset?Location=US&prefer_bqstorage_client=false",
        {
            "credentials": credentials,
            "bqstorage_client": storage,
            "default_query_job_config": config,
            "prefer_bqstorage_client": True,
        },
    )
    client.assert_called_once()
    kwargs = client.call_args.kwargs
    assert kwargs["project"] == "billing-project"
    assert kwargs["location"] == "US"
    assert kwargs["credentials"] is credentials
    assert (
        kwargs["default_query_job_config"].default_dataset
        == "billing-project.Some_Dataset"
    )
    assert kwargs["default_query_job_config"].use_query_cache is False
    assert config.default_dataset is None
    connect.assert_called_once_with(
        client=client.return_value,
        bqstorage_client=storage,
        prefer_bqstorage_client=True,
    )
    assert conn is connect.return_value
    assert close is not None
    close()
    conn.close.assert_called_once()
    client.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        ("bigquery://user:password@project/dataset", "expected bigquery"),
        ("bigquery://project:443/dataset", "expected bigquery"),
        ("bigquery://project/dataset/extra", "expected bigquery"),
        ("bigquery://project/dataset#fragment", "expected bigquery"),
        ("bigquery:///project?unknown=value", "no connect option.*unknown"),
        ("bigquery:///project?Location=US&location=EU", "duplicate"),
        ("bigquery://project/dataset?DatasetId=other", "single valid dataset"),
    ],
)
def test_bigquery_python_uri_rejects_unsupported_input(uri: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bridge._connect_bigquery(uri, {})


@pytest.mark.parametrize("use_wheel", [False, True])
def test_bigquery_adbc_adapts_only_the_resolved_wheel(
    use_wheel: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "wheel.so"
    manifest_library = tmp_path / "manifest.so"
    wheel.touch()
    manifest_library.touch()
    monkeypatch.setattr(bridge, "probe", lambda kind: ("1.11.0", str(wheel)))
    database, connection = Mock(), Mock()
    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager",
        SimpleNamespace(AdbcConnection=connection, AdbcDatabase=database),
    )
    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager._lib",
        SimpleNamespace(
            ArrowArrayStreamHandle=SimpleNamespace(__arrow_c_stream__=None)
        ),
    )
    uri = "bigquery:///project?Location=US"
    driver = str(wheel if use_wheel else manifest_library)
    options = [("uri", uri)]
    prefix = "adbc.bigquery.sql."
    session = bridge.adbc_connect(driver, None, options, [])
    if use_wheel:
        database.assert_called_once_with(
            driver=driver,
            **{prefix + "project_id": "project", prefix + "location": "US"},
        )
        assert session._execute_options == {prefix + "query.default_project_id": ""}
    else:
        database.assert_called_once_with(driver=driver, uri=uri)
        assert session._execute_options == {}
    session.close()


def test_bigquery_adbc_wheel_options_preserve_explicit_settings() -> None:
    prefix = "adbc.bigquery.sql."
    database = {prefix + "location": "EU"}
    statement = {prefix + "query.disable_query_cache": "true"}
    bridge._bigquery_wheel_options(
        "bigquery://project/Some_Dataset?Location=US", database, statement
    )
    assert database == {
        prefix + "project_id": "project",
        prefix + "dataset_id": "Some_Dataset",
        prefix + "location": "EU",
    }
    assert statement == {prefix + "query.disable_query_cache": "true"}


@pytest.mark.parametrize("driver", ["auto", "adbc", "path"])
@pytest.mark.parametrize(
    "destination_key",
    [
        None,
        "adbc.bigquery.sql.query.destination_table",
        "bigquery.query.destination_table",
    ],
)
def test_bigquery_adbc_scan_checks_statement_destination(
    driver: str,
    destination_key: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("sqlglot")
    library = tmp_path / "bigquery-driver.so"
    library.touch()
    probe = bridge.probe
    monkeypatch.setattr(
        bridge,
        "probe",
        lambda kind: (
            ("1.11.0", str(library)) if kind == "adbc:bigquery" else probe(kind)
        ),
    )
    table = pa.table({"id": pa.array([], type=pa.int64())})
    statement = Mock()
    statement.execute_query.side_effect = lambda: (table.to_reader(), 0)
    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager",
        SimpleNamespace(
            AdbcConnection=Mock(),
            AdbcDatabase=Mock(),
            AdbcStatement=Mock(return_value=statement),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager._lib",
        SimpleNamespace(ArrowArrayStreamHandle=pa.RecordBatchReader),
    )

    execute_options = {"adbc.bigquery.sql.query.write_disposition": "WRITE_TRUNCATE"}
    if destination_key is not None:
        execute_options[destination_key] = "billing-project.dataset.artificial"
    scan = pl.scan_database(
        "SELECT 17 AS id",
        "bigquery:///billing-project",
        driver={"auto": None, "adbc": "adbc", "path": str(library)}[driver],
        partition_on=None,
        execute_options=execute_options,
    )
    if destination_key is None:
        assert scan.collect_schema() == pl.Schema({"id": pl.Int64})
        statement.execute_query.assert_called_once()
    else:
        with pytest.raises(pl.exceptions.ComputeError, match="BigQuery destination"):
            scan.collect_schema()
        statement.set_sql_query.assert_not_called()
        statement.execute_query.assert_not_called()


@pytest.mark.parametrize("destination", ["", "billing-project.dataset.artificial"])
def test_bigquery_adbc_factory_checks_effective_statement_destination(
    destination: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("sqlglot")
    driver = pytest.importorskip("adbc_driver_bigquery.dbapi")
    manager = pytest.importorskip("adbc_driver_manager.dbapi")
    table = pa.table({"id": [17]})
    execute = Mock()

    def statement(_connection: Any) -> Mock:
        stmt = Mock()
        options: dict[str, str] = {}
        stmt.set_options.side_effect = lambda **kwargs: options.update(kwargs)
        stmt.get_option.side_effect = lambda key: options.get(key, "")
        return stmt

    monkeypatch.setattr(manager._lib, "AdbcStatement", statement)
    monkeypatch.setattr(
        manager.Connection, "adbc_get_info", lambda self: {"vendor_name": "BigQuery"}
    )
    monkeypatch.setattr(
        driver,
        "connect",
        lambda *, db_kwargs, autocommit: manager.Connection(
            Mock(), Mock(), autocommit=autocommit
        ),
    )

    def execute_cursor(self: Any, sql: str) -> Any:
        execute(sql)
        return self

    monkeypatch.setattr(manager.Cursor, "execute", execute_cursor)
    monkeypatch.setattr(
        manager.Cursor, "fetch_record_batch", lambda self: table.to_reader()
    )

    def connect() -> Any:
        conn = driver.connect(
            db_kwargs={"adbc.bigquery.sql.project_id": "billing-project"},
            autocommit=True,
        )
        cursor = conn.cursor
        statement_options = {
            "adbc.bigquery.sql.query.write_disposition": "WRITE_TRUNCATE"
        }
        if destination:
            statement_options["adbc.bigquery.sql.query.destination_table"] = destination
        conn.cursor = partial(
            cursor,
            adbc_stmt_kwargs=statement_options,
        )
        return conn

    scan = pl.scan_database(
        "SELECT 17 AS id", connect, partition_on=None, dialect="bigquery"
    )
    if destination:
        with pytest.raises(pl.exceptions.ComputeError, match="BigQuery destination"):
            scan.collect_schema()
        execute.assert_not_called()
    else:
        assert scan.collect_schema() == pl.Schema({"id": pl.Int64})
        execute.assert_called_once()


_BIGQUERY_ADBC_ROW = {"id": 17, "geo": "POINT(1 2)", "doc": '{"x":1}', "label": "a"}
_BIGQUERY_ADBC_SCHEMA = {
    "id": pl.Int64,
    "geo": pl.String,
    "doc": pl.String,
    "label": pl.String,
}


def _bigquery_adbc_table(pa: Any, *, empty: bool = False, probe: bool = False) -> Any:
    # ADBC can expose a native GEOGRAPHY or JSON as the same Arrow type as STRING.
    return pa.table(
        {
            name: pa.array(
                [] if empty else [None if probe else value],
                type=pa.int64() if name == "id" else pa.string(),
            )
            for name, value in _BIGQUERY_ADBC_ROW.items()
        }
    )


def _check_bigquery_adbc_text_filters(scan: pl.LazyFrame, sqls: list[str]) -> None:
    for name in ("geo", "doc", "label"):
        value = _BIGQUERY_ADBC_ROW[name]
        sqls.clear()
        assert scan.filter(pl.col(name) == value).select(name).collect().rows() == [
            (value,)
        ]
        assert any(sql.startswith(f"SELECT `{name}`") for sql in sqls)
        assert not any(f"WHERE `{name}`" in sql for sql in sqls), sqls

    sqls.clear()
    assert scan.filter(pl.col("id") == 17).select("id").collect().rows() == [(17,)]
    assert any("WHERE `id` = 17" in sql for sql in sqls), sqls


@pytest.mark.parametrize("empty", [False, True])
def test_bigquery_adbc_wheel_recovers_schema_from_empty_arrow_probe(
    empty: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("sqlglot")
    library = tmp_path / "wheel.so"
    library.touch()
    probe = bridge.probe
    monkeypatch.setattr(
        bridge,
        "probe",
        lambda kind: (
            ("1.11.0", str(library)) if kind == "adbc:bigquery" else probe(kind)
        ),
    )
    sqls: list[str] = []

    def statement(connection: Any) -> Mock:
        stmt = Mock()

        def execute_query() -> tuple[Any, int]:
            sql = stmt.set_sql_query.call_args.args[0]
            sqls.append(sql)
            if "SELECT _pl_record.*" in sql:
                assert "WHERE 1 = 0" in sql
                table = _bigquery_adbc_table(pa, probe=True)
            elif "WHERE 1 = 0" in sql or empty:
                table = pa.table({})
            else:
                table = _bigquery_adbc_table(pa)
            return table.to_reader(), table.num_rows

        stmt.execute_query.side_effect = execute_query
        return stmt

    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager",
        SimpleNamespace(
            AdbcConnection=Mock(), AdbcDatabase=Mock(), AdbcStatement=statement
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager._lib",
        SimpleNamespace(ArrowArrayStreamHandle=pa.RecordBatchReader),
    )
    scan = pl.scan_database(
        "SELECT * FROM fixture",
        "bigquery:///project",
        driver=str(library),
        partition_on=None,
    )
    assert scan.collect_schema() == pl.Schema(_BIGQUERY_ADBC_SCHEMA)
    assert scan.collect().to_dicts() == ([] if empty else [_BIGQUERY_ADBC_ROW])
    assert any("SELECT _pl_record.*" in sql for sql in sqls)
    if not empty:
        _check_bigquery_adbc_text_filters(scan, sqls)


@pytest.mark.parametrize("dialect", [None, "bigquery"])
@pytest.mark.parametrize("empty", [False, True])
def test_bigquery_adbc_factory_recovers_schema_from_empty_arrow_probe(
    dialect: str | None, empty: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("sqlglot")
    monkeypatch.setitem(
        sys.modules,
        "adbc_driver_manager",
        SimpleNamespace(
            __spec__=ModuleSpec("adbc_driver_manager", loader=None),
            __version__="1.13.0",
        ),
    )
    sqls: list[str] = []

    class Cursor:
        def execute(self, sql: str) -> None:
            sqls.append(sql)
            if "SELECT _pl_record.*" in sql:
                self.result = _bigquery_adbc_table(pa, probe=True)
            elif "WHERE 1 = 0" in sql:
                self.result = pa.table({})
            else:
                self.result = _bigquery_adbc_table(pa, empty=empty)

        def fetch_record_batch(self) -> Any:
            return self.result.to_reader()

        def close(self) -> None:
            pass

    class Connection:
        __module__ = "adbc_driver_manager.dbapi"

        def adbc_get_info(self) -> dict[str, str]:
            return {"vendor_name": "BigQuery"}

        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            pass

    scan = pl.scan_database(
        "SELECT * FROM fixture",
        lambda: Connection(),
        dialect=dialect,
        partition_on=None,
    )
    assert scan.collect_schema() == pl.Schema(_BIGQUERY_ADBC_SCHEMA)
    assert scan.collect().to_dicts() == ([] if empty else [_BIGQUERY_ADBC_ROW])
    assert any("SELECT _pl_record.*" in sql for sql in sqls)
    if not empty:
        _check_bigquery_adbc_text_filters(scan, sqls)
