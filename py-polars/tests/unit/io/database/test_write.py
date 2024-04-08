from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import DbWriteEngine


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    "engine",
    [
        "sqlalchemy",
        pytest.param(
            "adbc",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc not available on Windows or <= Python 3.8",
            ),
        ),
    ],
)
class TestWriteDatabase:
    """Database write tests that share common pytest/parametrize options."""

    def test_write_database_create(self, engine: DbWriteEngine, tmp_path: Path) -> None:
        """Test basic database table creation."""
        df = pl.DataFrame(
            {
                "id": [1234, 5678],
                "name": ["misc", "other"],
                "value": [1000.0, -9999.0],
            }
        )
        tmp_path.mkdir(exist_ok=True)
        test_db = str(tmp_path / f"test_{engine}.db")
        test_db_uri = f"sqlite:///{test_db}"
        table_name = "test_create"

        assert (
            df.write_database(
                table_name=table_name,
                connection=test_db_uri,
                engine=engine,
            )
            == 2
        )
        result = pl.read_database(
            query=f"SELECT * FROM {table_name}",
            connection=create_engine(test_db_uri),
        )
        assert_frame_equal(result, df)

    def test_write_database_append_replace(
        self, engine: DbWriteEngine, tmp_path: Path
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
        test_db = str(tmp_path / f"test_{engine}.db")
        test_db_uri = f"sqlite:///{test_db}"
        table_name = f"test_append_{engine}"

        assert (
            df.write_database(
                table_name=table_name,
                connection=test_db_uri,
                engine=engine,
            )
            == 3
        )
        with pytest.raises(Exception):  # noqa: B017
            df.write_database(
                table_name=table_name,
                connection=test_db_uri,
                if_table_exists="fail",
                engine=engine,
            )

        assert (
            df.write_database(
                table_name=table_name,
                connection=test_db_uri,
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
                connection=test_db_uri,
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

    def test_write_database_create_quoted_tablename(
        self, engine: DbWriteEngine, tmp_path: Path
    ) -> None:
        """Test parsing/handling of quoted database table names."""
        df = pl.DataFrame({"col x": [100, 200, 300], "col y": ["a", "b", "c"]})

        tmp_path.mkdir(exist_ok=True)
        test_db = str(tmp_path / f"test_{engine}.db")
        test_db_uri = f"sqlite:///{test_db}"

        # table name requires quoting, and is qualified with the implicit 'main' schema
        qualified_table_name = f'main."test-append-{engine}"'

        assert (
            df.write_database(
                table_name=qualified_table_name,
                connection=test_db_uri,
                engine=engine,
            )
            == 3
        )
        assert (
            df.write_database(
                table_name=qualified_table_name,
                connection=test_db_uri,
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

    def test_write_database_errors(self, engine: DbWriteEngine, tmp_path: Path) -> None:
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
