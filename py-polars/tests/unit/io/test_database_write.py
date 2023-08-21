from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import DbWriteEngine, DbWriteMode


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["misc", "other"],
            "value": [100.0, -99.0],
            "date": ["2020-01-01", "2021-12-31"],
        }
    )


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("engine", "mode"),
    [
        pytest.param(
            "adbc",
            "create",
            id="adbc_create",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
        ),
        pytest.param(
            "adbc",
            "append",
            id="adbc_append",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
        ),
        pytest.param(
            "sqlalchemy",
            "create",
            id="sa_create",
        ),
        pytest.param(
            "sqlalchemy",
            "append",
            id="sa_append",
        ),
    ],
)
def test_write_database(
    engine: DbWriteEngine, mode: DbWriteMode, sample_df: pl.DataFrame, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)
    tmp_db = f"test_{engine}.db"
    test_db = str(tmp_path / tmp_db)

    # note: test a table name that requires quotes to ensure that we handle
    # it correctly (also supply an explicit db schema with/without quotes)
    tbl_name = '"test-data"'

    sample_df.write_database(
        table_name=f"main.{tbl_name}",
        connection=f"sqlite:///{test_db}",
        if_exists="replace",
        engine=engine,
    )
    if mode == "append":
        sample_df.write_database(
            table_name=f'"main".{tbl_name}',
            connection=f"sqlite:///{test_db}",
            if_exists="append",
            engine=engine,
        )
        sample_df = pl.concat([sample_df, sample_df])

    result = pl.read_database_uri(f"SELECT * FROM {tbl_name}", f"sqlite:///{test_db}")
    sample_df = sample_df.with_columns(pl.col("date").cast(pl.Utf8))
    assert_frame_equal(sample_df, result)

    # check that some invalid parameters raise errors
    for invalid_params in (
        {"table_name": "w.x.y.z"},
        {"if_exists": "crunk", "table_name": f"main.{tbl_name}"},
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            sample_df.write_database(
                connection=f"sqlite:///{test_db}",
                engine=engine,
                **invalid_params,  # type: ignore[arg-type]
            )
