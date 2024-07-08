from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

import polars as pl
from polars.io.database._utils import _open_adbc_connection
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import DbWriteEngine


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("engine", "uri_connection"),
    [
        ("sqlalchemy", True),
        ("sqlalchemy", False),
        pytest.param(
            "adbc",
            True,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc not available on Windows or <= Python 3.8",
            ),
        ),
        pytest.param(
            "adbc",
            False,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc not available on Windows or <= Python 3.8",
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
            return create_engine(uri)
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
        result = pl.read_database(
            query=f"SELECT * FROM {table_name}",
            connection=create_engine(test_db_uri),
        )
        assert_frame_equal(result, df)

        if hasattr(conn, "close"):
            conn.close()

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
        result = pl.read_database(
            query=f"SELECT * FROM {table_name}",
            connection=create_engine(test_db_uri),
        )
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
        result = pl.read_database(
            query=f"SELECT * FROM {table_name}",
            connection=create_engine(test_db_uri),
        )
        assert_frame_equal(result, pl.concat([df, df[:2]]))

        if engine == "adbc" and not uri_connection:
            assert conn._closed is False

        if hasattr(conn, "close"):
            conn.close()

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
        result = pl.read_database(
            query=f"SELECT * FROM {qualified_table_name}",
            connection=create_engine(test_db_uri),
        )
        assert_frame_equal(result, df)

        if engine == "adbc" and not uri_connection:
            assert conn._closed is False

        if hasattr(conn, "close"):
            conn.close()

    def test_write_database_errors(
        self, engine: DbWriteEngine, uri_connection: bool, tmp_path: Path
    ) -> None:
        """Confirm that expected errors are raised."""
        df = pl.DataFrame({"colx": [1, 2, 3]})

        with pytest.raises(
            ValueError, match="`table_name` appears to be invalid: 'w.x.y.z'"
        ):
            df.write_database(
                connection="sqlite:///:memory:",
                table_name="w.x.y.z",
                engine=engine,
            )

        with pytest.raises(
            ValueError,
            match="`if_table_exists` must be one of .* got 'do_something'",
        ):
            df.write_database(
                connection="sqlite:///:memory:",
                table_name="main.test_errs",
                if_table_exists="do_something",  # type: ignore[arg-type]
                engine=engine,
            )

        with pytest.raises(
            TypeError,
            match="unrecognised connection type",
        ):
            df.write_database(connection=True, table_name="misc")  # type: ignore[arg-type]


@pytest.mark.write_disk()
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
    engine = create_engine(test_db_uri, poolclass=NullPool)
    with Session(engine) as session:
        df.write_database(table_name, session)
        session.commit()

    with Session(engine) as session:
        result = pl.read_database(
            query=f"select * from {table_name}", connection=session
        )

    assert_frame_equal(result, df)


@pytest.mark.write_disk()
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
    engine = create_engine(test_db_uri, poolclass=NullPool)
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


@pytest.mark.write_disk()
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
    engine = create_engine(test_db_uri, poolclass=NullPool)
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
