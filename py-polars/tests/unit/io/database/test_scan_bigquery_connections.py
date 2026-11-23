from __future__ import annotations

import sys
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import polars as pl
from polars._utils.various import parse_version

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
    from google.cloud.bigquery import QueryJobConfig, SchemaField

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
    from google.cloud.bigquery import QueryJobConfig, dbapi

    def unexpected_query(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("scan issued a query with a destination table")

    for method in ("query", "query_and_wait"):
        monkeypatch.setattr(type(bigquery_client), method, unexpected_query)
    cfg = QueryJobConfig(
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
    from google.cloud.bigquery import SchemaField

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

        factory_scan = pl.scan_database(
            "SELECT 17 AS id", lambda: engine, partition_on=None
        )
        with pytest.raises(
            pl.exceptions.ComputeError, match=r"BigQuery.*pool_pre_ping"
        ):
            factory_scan.collect_schema()

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


def test_bigquery_sqlalchemy_session_job_config(
    bigquery_engine: Any, bigquery_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    orm = pytest.importorskip("sqlalchemy.orm")
    pytest.importorskip("sqlglot")
    if parse_version(sa.__version__) < (2, 1):
        pytest.skip("Session-wide execution options require SQLAlchemy 2.1")
    from google.cloud.bigquery import QueryJobConfig, SchemaField

    def scan(session: Any) -> pl.LazyFrame:
        return pl.scan_database(
            "SELECT :answer AS id",
            session,
            params={"answer": 17},
            schema={"id": pl.Int64},
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

    def harmless(_sql: str, *, job_config: Any, **_kwargs: Any) -> _GoogleRows:
        assert job_config.maximum_bytes_billed == 1024
        return _GoogleRows([SchemaField("id", "INTEGER")], [(17,)])

    monkeypatch.setattr(bigquery_client, "query_and_wait", harmless)
    safe = QueryJobConfig(maximum_bytes_billed=1024)
    with orm.Session(bigquery_engine, execution_options={"job_config": safe}) as s:
        assert scan(s).collect().rows() == [(17,)]


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
                table = pa.table({"id": pa.array([None], type=pa.int64())})
            elif "WHERE 1 = 0" in sql or empty:
                table = pa.table({})
            else:
                table = pa.table({"id": pa.array([17], type=pa.int64())})
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
        "SELECT 17 AS id",
        "bigquery:///project",
        driver=str(library),
        partition_on=None,
    )
    assert scan.collect_schema() == pl.Schema({"id": pl.Int64})
    assert scan.collect().rows() == ([] if empty else [(17,)])
    assert any("SELECT _pl_record.*" in sql for sql in sqls)


@pytest.mark.parametrize("dialect", [None, "bigquery"])
@pytest.mark.parametrize("empty", [False, True])
def test_bigquery_adbc_factory_recovers_schema_from_empty_arrow_probe(
    dialect: str | None, empty: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("sqlglot")
    monkeypatch.setitem(
        sys.modules, "adbc_driver_manager", SimpleNamespace(__version__="1.13.0")
    )
    sqls: list[str] = []

    class Cursor:
        def execute(self, sql: str) -> None:
            sqls.append(sql)
            if "SELECT _pl_record.*" in sql:
                self.result = pa.table({"id": pa.array([None], type=pa.int64())})
            elif "WHERE 1 = 0" in sql:
                self.result = pa.table({})
            else:
                self.result = pa.table(
                    {"id": pa.array([] if empty else [17], type=pa.int64())}
                )

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
        "SELECT 17 AS id", lambda: Connection(), dialect=dialect, partition_on=None
    )
    assert scan.collect_schema() == pl.Schema({"id": pl.Int64})
    assert scan.collect().rows() == ([] if empty else [(17,)])
    assert any("SELECT _pl_record.*" in sql for sql in sqls)
