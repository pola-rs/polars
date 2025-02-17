from __future__ import annotations

import asyncio
import sys
import time
from functools import partial
from typing import Any, Callable

import pytest

import polars as pl
from polars.dependencies import gevent
from polars.exceptions import ColumnNotFoundError

pytestmark = pytest.mark.slow()


async def _aio_collect_async(raises: bool = False) -> pl.DataFrame:
    lf = (
        pl.LazyFrame(
            {
                "a": ["a", "b", "a", "b", "b", "c"],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )
        .group_by("a", maintain_order=True)
        .agg(pl.all().sum())
    )
    if raises:
        lf = lf.select(pl.col("foo_bar"))
    return await lf.collect_async()


async def _aio_collect_all_async(raises: bool = False) -> list[pl.DataFrame]:
    lf = (
        pl.LazyFrame(
            {
                "a": ["a", "b", "a", "b", "b", "c"],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )
        .group_by("a", maintain_order=True)
        .agg(pl.all().sum())
    )
    if raises:
        lf = lf.select(pl.col("foo_bar"))

    lf2 = pl.LazyFrame({"a": [1, 2], "b": [1, 2]}).group_by("a").sum()

    return await pl.collect_all_async([lf, lf2])


_aio_collect = pytest.mark.parametrize(
    ("collect", "raises"),
    [
        (_aio_collect_async, None),
        (_aio_collect_all_async, None),
        (partial(_aio_collect_async, True), ColumnNotFoundError),
        (partial(_aio_collect_all_async, True), ColumnNotFoundError),
    ],
)


def _aio_run(coroutine: Any, raises: Exception | None = None) -> None:
    if raises is not None:
        with pytest.raises(raises):  # type: ignore[call-overload]
            asyncio.run(coroutine)
    else:
        assert len(asyncio.run(coroutine)) > 0


@_aio_collect
def test_collect_async_switch(
    collect: Callable[[], Any],
    raises: Exception | None,
) -> None:
    async def main() -> Any:
        df = collect()
        await asyncio.sleep(0.3)
        return await df

    _aio_run(main(), raises)


@_aio_collect
def test_collect_async_task(
    collect: Callable[[], Any], raises: Exception | None
) -> None:
    async def main() -> Any:
        df = asyncio.create_task(collect())
        await asyncio.sleep(0.3)
        return await df

    _aio_run(main(), raises)


def _gevent_collect_async(raises: bool = False) -> Any:
    lf = (
        pl.LazyFrame(
            {
                "a": ["a", "b", "a", "b", "b", "c"],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )
        .group_by("a", maintain_order=True)
        .agg(pl.all().sum())
    )
    if raises:
        lf = lf.select(pl.col("foo_bar"))
    return lf.collect_async(gevent=True)


def _gevent_collect_all_async(raises: bool = False) -> Any:
    lf = (
        pl.LazyFrame(
            {
                "a": ["a", "b", "a", "b", "b", "c"],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )
        .group_by("a", maintain_order=True)
        .agg(pl.all().sum())
    )
    if raises:
        lf = lf.select(pl.col("foo_bar"))
    return pl.collect_all_async([lf], gevent=True)


_gevent_collect = pytest.mark.parametrize(
    ("get_result", "raises"),
    [
        (_gevent_collect_async, None),
        (_gevent_collect_all_async, None),
        (partial(_gevent_collect_async, True), ColumnNotFoundError),
        (partial(_gevent_collect_all_async, True), ColumnNotFoundError),
    ],
)


def _gevent_run(callback: Callable[[], Any], raises: Exception | None = None) -> None:
    if raises is not None:
        with pytest.raises(raises):  # type: ignore[call-overload]
            callback()
    else:
        assert len(callback()) > 0


@_gevent_collect
def test_gevent_collect_async_without_hub(
    get_result: Callable[[], Any], raises: Exception | None
) -> None:
    def main() -> Any:
        return get_result().get()

    _gevent_run(main, raises)


@_gevent_collect
def test_gevent_collect_async_with_hub(
    get_result: Callable[[], Any], raises: Exception | None
) -> None:
    _hub = gevent.get_hub()

    def main() -> Any:
        return get_result().get()

    _gevent_run(main, raises)


@pytest.mark.skipif(sys.platform == "win32", reason="May time out on Windows")
@_gevent_collect
def test_gevent_collect_async_switch(
    get_result: Callable[[], Any], raises: Exception | None
) -> None:
    def main() -> Any:
        result = get_result()
        gevent.sleep(0.1)
        return result.get(block=False, timeout=3)

    _gevent_run(main, raises)


@_gevent_collect
def test_gevent_collect_async_no_switch(
    get_result: Callable[[], Any], raises: Exception | None
) -> None:
    def main() -> Any:
        result = get_result()
        time.sleep(1)
        return result.get(block=False, timeout=None)

    _gevent_run(main, raises)


@_gevent_collect
def test_gevent_collect_async_spawn(
    get_result: Callable[[], Any], raises: Exception | None
) -> None:
    def main() -> Any:
        result_greenlet = gevent.spawn(get_result)
        gevent.spawn(gevent.sleep, 0.1)
        return result_greenlet.get().get()

    _gevent_run(main, raises)
