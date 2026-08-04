from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.orm import Session

import polars as pl
from polars._utils.various import parse_version
from polars.io.database._utils import _open_adbc_connection
from polars.testing import assert_frame_equal
from tests.unit.io.database.conftest import close_connections, create_sqlite_engine

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import DbWriteEngine


def _read_via_uri(query: str, uri: str) -> pl.DataFrame:
    """Read a query result via a throwaway engine, disposing it afterwards."""
    engine = create_sqlite_engine(uri)
    try:
        return pl.read_database(query=query, connection=engine)
    finally:
        engine.dispose()


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("engine", "uri_connection"),
    [
        ("sqlalchemy", True),
        ("sqlalchemy", False),
        pytest.param(
            "adbc",
            True,
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="adbc not available on Windows",
            ),
        ),
        pytest.param(
            "adbc",
            False,
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="adbc not available on Windows",
            ),
        ),
    ],
)
class TestWriteDatabase:
    """Database write tests that share common pytest/parametrize options."""

    @staticmethod
    def _get_connection(uri: str, engine: DbWriteEngine, uri_connection: bool) -> Any:
        if uri_connection:
            return uri
        elif engine == "sqlalchemy":
            return create_sqlite_engine(uri)
        else:
            return _open_adbc_connection(uri)

    def test_write_database_create(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        """Test basic database table creation."""
        df = pl.DataFrame(
            {
                "id": [1234, 5678],
                "name": ["misc", "other"],
                "value": [1000.0, -9999.0],
            }
        )
        tmp_path.mkdir(exist_ok=True)
        test_db_uri = f"sqlite:///{tmp_path}/test_create_{int(uri_connection)}.db"

        table_name = "test_create"
        conn = self._get_connection(test_db_uri, engine, uri_connection)

        assert (
            df.write_database(
                table_name=table_name,
                connection=conn,
                engine=engine,
            )
            == 2
        )
        result = _read_via_uri(f"SELECT * FROM {table_name}", test_db_uri)
        assert_frame_equal(result, df)

        close_connections(conn)

    def test_write_database_append_replace(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        """Test append/replace ops against existing database table."""
        df = pl.DataFrame(
            {
                "key": ["xx", "yy", "zz"],
                "value": [123, None, 789],
                "other": [5.5, 7.0, None],
            }
        )
        tmp_path.mkdir(exist_ok=True)
        test_db_uri = f"sqlite:///{tmp_path}/test_append_{int(uri_connection)}.db"

        table_name = "test_append"
        conn = self._get_connection(test_db_uri, engine, uri_connection)

        assert (
            df.write_database(
                table_name=table_name,
                connection=conn,
                engine=engine,
            )
            == 3
        )
        with pytest.raises(Exception):  # noqa: B017
            df.write_database(
                table_name=table_name,
                connection=conn,
                if_table_exists="fail",
                engine=engine,
            )

        assert (
            df.write_database(
                table_name=table_name,
                connection=conn,
                if_table_exists="replace",
                engine=engine,
            )
            == 3
        )
        result = _read_via_uri(f"SELECT * FROM {table_name}", test_db_uri)
        assert_frame_equal(result, df)

        assert (
            df[:2].write_database(
                table_name=table_name,
                connection=conn,
                if_table_exists="append",
                engine=engine,
            )
            == 2
        )
        result = _read_via_uri(f"SELECT * FROM {table_name}", test_db_uri)
        assert_frame_equal(result, pl.concat([df, df[:2]]))

        if engine == "adbc" and not uri_connection:
            assert conn._closed is False

        close_connections(conn)

    def test_write_database_append_creates_missing_table(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        """`append` should create table when one does not already exist."""
        if engine == "adbc":
            adbc_driver_manager = pytest.importorskip("adbc_driver_manager")
            if parse_version(getattr(adbc_driver_manager, "__version__", "0.0")) < (
                0,
                7,
            ):
                pytest.skip("adbc-driver-manager < 0.7.0 has no create_append mode")

        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
            }
        )
        tmp_path.mkdir(exist_ok=True)
        test_db_uri = (
            f"sqlite:///{tmp_path}/test_append_create_{int(uri_connection)}.db"
        )

        table_name = "test_append_create"
        conn = self._get_connection(test_db_uri, engine, uri_connection)

        assert (
            df.write_database(
                table_name=table_name,
                connection=conn,
                if_table_exists="append",
                engine=engine,
            )
            == 3
        )
        result = _read_via_uri(f"SELECT * FROM {table_name}", test_db_uri)
        assert_frame_equal(result, df)

        close_connections(conn)

    def test_write_database_append_existing_table_no_create(
        self,
        engine: DbWriteEngine,
        uri_connection: bool,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Appending to an existing table shouldn't require CREATE (ref: #27886)."""
        if engine != "adbc":
            pytest.skip("mode selection only applies to the 'adbc' engine")

        adbc_driver_manager = pytest.importorskip("adbc_driver_manager")
        if parse_version(getattr(adbc_driver_manager, "__version__", "0.0")) < (0, 7):
            pytest.skip("adbc-driver-manager < 0.7.0 always uses 'append' mode")

        import adbc_driver_manager.dbapi as adbc_dbapi

        # record the ingest mode used for each write, so we can assert that appending
        # to an *existing* table uses plain 'append' (INSERT only) rather than
        # 'create_append' (which additionally requires CREATE TABLE privileges)
        ingest_modes: list[str | None] = []
        original_ingest = adbc_dbapi.Cursor.adbc_ingest

        def _record_mode(self: Any, *args: Any, **kwargs: Any) -> Any:
            ingest_modes.append(kwargs.get("mode"))
            return original_ingest(self, *args, **kwargs)

        monkeypatch.setattr(adbc_dbapi.Cursor, "adbc_ingest", _record_mode)

        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        tmp_path.mkdir(exist_ok=True)
        test_db_uri = (
            f"sqlite:///{tmp_path}/test_append_existing_{int(uri_connection)}.db"
        )
        table_name = "test_append_existing"
        conn = self._get_connection(test_db_uri, engine, uri_connection)

        # first append creates the missing table...
        df.write_database(
            table_name=table_name,
            connection=conn,
            if_table_exists="append",
            engine=engine,
        )
        # ...second append targets the now-created table
        df.write_database(
            table_name=table_name,
            connection=conn,
            if_table_exists="append",
            engine=engine,
        )
        assert ingest_modes == ["create_append", "append"]

        result = _read_via_uri(f"SELECT * FROM {table_name}", test_db_uri)
        assert_frame_equal(result, pl.concat([df, df]))
        close_connections(conn)

    def test_write_database_create_quoted_tablename(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        """Test parsing/handling of quoted database table names."""
        df = pl.DataFrame(
            {
                "col x": [100, 200, 300],
                "col y": ["a", "b", "c"],
            }
        )
        tmp_path.mkdir(exist_ok=True)
        test_db_uri = f"sqlite:///{tmp_path}/test_create_quoted.db"

        # table name has some special chars, so requires quoting, and
        # is explicitly qualified with the sqlite 'main' schema
        qualified_table_name = f'main."test-append-{engine}-{int(uri_connection)}"'
        conn = self._get_connection(test_db_uri, engine, uri_connection)

        assert (
            df.write_database(
                table_name=qualified_table_name,
                connection=conn,
                engine=engine,
            )
            == 3
        )
        assert (
            df.write_database(
                table_name=qualified_table_name,
                connection=conn,
                if_table_exists="replace",
                engine=engine,
            )
            == 3
        )
        result = _read_via_uri(f"SELECT * FROM {qualified_table_name}", test_db_uri)
        assert_frame_equal(result, df)

        if engine == "adbc" and not uri_connection:
            assert conn._closed is False

        close_connections(conn)

    def test_write_database_errors(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        """Confirm that expected errors are raised."""
        df = pl.DataFrame({"colx": [1, 2, 3]})

        with pytest.raises(
            ValueError, match=r"`table_name` appears to be invalid: 'w.x.y.z'"
        ):
            df.write_database(
                connection="sqlite:///:memory:",
                table_name="w.x.y.z",
                engine=engine,
            )

        with pytest.raises(
            ValueError,
            match=r"`if_table_exists` must be one of .* got 'do_something'",
        ):
            df.write_database(
                connection="sqlite:///:memory:",
                table_name="main.test_errs",
                if_table_exists="do_something",  # type: ignore[arg-type]
                engine=engine,
            )

        with pytest.raises(
            TypeError,
            match=r"unrecognised connection type.*",
        ):
            df.write_database(connection=True, table_name="misc", engine=engine)  # type: ignore[arg-type]

    def test_write_database_adbc_missing_driver_error(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        # Skip for sqlalchemy
        if engine == "sqlalchemy":
            return
        df = pl.DataFrame({"colx": [1, 2, 3]})
        with pytest.raises(
            ModuleNotFoundError, match=r"ADBC 'adbc_driver_mysql' driver not detected."
        ):
            df.write_database(
                table_name="my_schema.my_table",
                connection="mysql:///:memory:",
                engine=engine,
            )


@pytest.mark.write_disk
def test_write_database_using_sa_session(tmp_path: str) -> None:
    df = pl.DataFrame(
        {
            "key": ["xx", "yy", "zz"],
            "value": [123, None, 789],
            "other": [5.5, 7.0, None],
        }
    )
    table_name = "test_sa_session"
    test_db_uri = f"sqlite:///{tmp_path}/test_sa_session.db"
    engine = create_sqlite_engine(test_db_uri)
    with Session(engine) as session:
        df.write_database(table_name, session)
        session.commit()

    with Session(engine) as session:
        result = pl.read_database(
            query=f"select * from {table_name}", connection=session
        )

    assert_frame_equal(result, df)


@pytest.mark.write_disk
@pytest.mark.parametrize("pass_connection", [True, False])
def test_write_database_sa_rollback(tmp_path: str, pass_connection: bool) -> None:
    df = pl.DataFrame(
        {
            "key": ["xx", "yy", "zz"],
            "value": [123, None, 789],
            "other": [5.5, 7.0, None],
        }
    )
    table_name = "test_sa_rollback"
    test_db_uri = f"sqlite:///{tmp_path}/test_sa_rollback.db"
    engine = create_sqlite_engine(test_db_uri)
    with Session(engine) as session:
        if pass_connection:
            conn = session.connection()
            df.write_database(table_name, conn)
        else:
            df.write_database(table_name, session)
        session.rollback()

    with Session(engine) as session:
        count = pl.read_database(
            query=f"select count(*) from {table_name}", connection=session
        ).item(0, 0)

    assert isinstance(count, int)
    assert count == 0


@pytest.mark.write_disk
@pytest.mark.parametrize("pass_connection", [True, False])
def test_write_database_sa_commit(tmp_path: str, pass_connection: bool) -> None:
    df = pl.DataFrame(
        {
            "key": ["xx", "yy", "zz"],
            "value": [123, None, 789],
            "other": [5.5, 7.0, None],
        }
    )
    table_name = "test_sa_commit"
    test_db_uri = f"sqlite:///{tmp_path}/test_sa_commit.db"
    engine = create_sqlite_engine(test_db_uri)
    with Session(engine) as session:
        if pass_connection:
            conn = session.connection()
            df.write_database(table_name, conn)
        else:
            df.write_database(table_name, session)
        session.commit()

    with Session(engine) as session:
        result = pl.read_database(
            query=f"select * from {table_name}", connection=session
        )

    assert_frame_equal(result, df)


@pytest.mark.skipif(sys.platform == "win32", reason="adbc not available on Windows")
def test_write_database_adbc_temporary_table() -> None:
    """Confirm that execution_options are passed along to create temporary tables."""
    df = pl.DataFrame({"colx": [1, 2, 3]})
    temp_tbl_name = "should_be_temptable"
    expected_temp_table_create_sql = (
        """CREATE TABLE "should_be_temptable" ("colx" INTEGER)"""
    )

    # test with sqlite in memory
    conn = _open_adbc_connection("sqlite:///:memory:")
    assert (
        df.write_database(
            temp_tbl_name,
            connection=conn,
            if_table_exists="fail",
            engine_options={"temporary": True},
        )
        == 3
    )
    temp_tbl_sql_df = pl.read_database(
        "select sql from sqlite_temp_master where type='table' and tbl_name = ?",
        connection=conn,
        execute_options={"parameters": [temp_tbl_name]},
    )
    assert temp_tbl_sql_df.shape[0] == 1, "no temp table created"
    actual_temp_table_create_sql = temp_tbl_sql_df["sql"][0]
    assert expected_temp_table_create_sql == actual_temp_table_create_sql

    close_connections(conn)
