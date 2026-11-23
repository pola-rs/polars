from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

bridge = pytest.importorskip("polars_database._bridge")


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
