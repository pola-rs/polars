from __future__ import annotations

import asyncio
from math import ceil
from typing import TYPE_CHECKING, Any, Iterable, overload

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine

import polars as pl
from polars._utils.various import parse_version
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

SURREAL_MOCK_DATA: list[dict[str, Any]] = [
    {
        "id": "item:8xj31jfpdkf9gvmxdxpi",
        "name": "abc",
        "tags": ["polars"],
        "checked": False,
    },
    {
        "id": "item:l59k19swv2adsv4q04cj",
        "name": "mno",
        "tags": ["async"],
        "checked": None,
    },
    {
        "id": "item:w831f1oyqnwztv5q03em",
        "name": "xyz",
        "tags": ["stroop", "wafel"],
        "checked": True,
    },
]


class MockSurrealConnection:
    """Mock SurrealDB connection/client object."""

    __module__ = "surrealdb"

    def __init__(self, url: str, mock_data: list[dict[str, Any]]) -> None:
        self._mock_data = mock_data.copy()
        self.url = url

    async def __aenter__(self) -> Any:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await self.close()

    async def close(self) -> None:
        pass

    async def connect(self) -> None:
        pass

    async def use(self, namespace: str, database: str) -> None:
        pass

    async def query(
        self, sql: str, vars: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [{"result": self._mock_data, "status": "OK", "time": "32.083µs"}]


@pytest.mark.skipif(
    parse_version(sqlalchemy.__version__) < (2, 0),
    reason="SQLAlchemy 2.0+ required for async tests",
)
def test_read_async(tmp_sqlite_db: Path) -> None:
    # confirm that we can load frame data from the core sqlalchemy async
    # primitives: AsyncConnection, AsyncEngine, and async_sessionmaker
    from sqlalchemy.ext.asyncio import async_sessionmaker

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_sqlite_db}")
    async_connection = async_engine.connect()
    async_session = async_sessionmaker(async_engine)
    async_session_inst = async_session()

    expected_frame = pl.DataFrame(
        {"id": [2, 1], "name": ["other", "misc"], "value": [-99.5, 100.0]}
    )
    async_conn: Any
    for async_conn in (
        async_engine,
        async_connection,
        async_session,
        async_session_inst,
    ):
        if async_conn in (async_session, async_session_inst):
            constraint, execute_opts = "", {}
        else:
            constraint = "WHERE value > :n"
            execute_opts = {"parameters": {"n": -1000}}

        df = pl.read_database(
            query=f"""
                SELECT id, name, value
                FROM test_data {constraint}
                ORDER BY id DESC
            """,
            connection=async_conn,
            execute_options=execute_opts,
        )
        assert_frame_equal(expected_frame, df)


async def _nested_async_test(tmp_sqlite_db: Path) -> pl.DataFrame:
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_sqlite_db}")
    return pl.read_database(
        query="SELECT id, name FROM test_data ORDER BY id",
        connection=async_engine.connect(),
    )


@pytest.mark.skipif(
    parse_version(sqlalchemy.__version__) < (2, 0),
    reason="SQLAlchemy 2.0+ required for async tests",
)
def test_read_async_nested(tmp_sqlite_db: Path) -> None:
    # this tests validates that we can handle nested async calls. without
    # the nested asyncio handling provided by `nest_asyncio` this test
    # would raise a RuntimeError

    expected_frame = pl.DataFrame({"id": [1, 2], "name": ["misc", "other"]})
    df = asyncio.run(_nested_async_test(tmp_sqlite_db))
    assert_frame_equal(expected_frame, df)


@overload
async def _surreal_query_as_frame(
    url: str, query: str, batch_size: None
) -> pl.DataFrame: ...


@overload
async def _surreal_query_as_frame(
    url: str, query: str, batch_size: int
) -> Iterable[pl.DataFrame]: ...


async def _surreal_query_as_frame(
    url: str, query: str, batch_size: int | None
) -> pl.DataFrame | Iterable[pl.DataFrame]:
    batch_params = (
        {"iter_batches": True, "batch_size": batch_size} if batch_size else {}
    )
    async with MockSurrealConnection(url=url, mock_data=SURREAL_MOCK_DATA) as client:
        await client.use(namespace="test", database="test")
        return pl.read_database(  # type: ignore[no-any-return,call-overload]
            query=query,
            connection=client,
            **batch_params,
        )


@pytest.mark.parametrize("batch_size", [None, 1, 2, 3, 4])
def test_surrealdb_fetchall(batch_size: int | None) -> None:
    df_expected = pl.DataFrame(SURREAL_MOCK_DATA)
    res = asyncio.run(
        _surreal_query_as_frame(
            url="ws://localhost:8000/rpc",
            query="SELECT * FROM item",
            batch_size=batch_size,
        )
    )
    if batch_size:
        frames = list(res)  # type: ignore[call-overload]
        n_mock_rows = len(SURREAL_MOCK_DATA)
        assert len(frames) == ceil(n_mock_rows / batch_size)
        assert_frame_equal(df_expected[:batch_size], frames[0])
    else:
        assert_frame_equal(df_expected, res)  # type: ignore[arg-type]
