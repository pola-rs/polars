from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.wrap import wrap_df

if TYPE_CHECKING:
    from polars import DataFrame
    from polars._plr import PyInProcessQuery


class InProcessQuery:
    """
    A placeholder for an in process query.

    This can be used to do something else while a query is running.
    The queries can be cancelled. You can peek if the query is finished,
    or you can await the result.
    """

    def __init__(self, ipq: PyInProcessQuery) -> None:
        self._inner = ipq

    def cancel(self) -> None:
        """
        Cancel the query, and wait until it has stopped.

        When this returns, nothing of the query runs any more: its tasks have
        ended and its sources and sinks have been closed (a database source, for
        example, has finished with its connection). This can block until the
        running work reaches a point where it can stop, such as the end of a
        batch a source is reading.

        Calling it again, or after the query has finished or its result was
        fetched, returns at once. Called from one of Polars' own threads, it only
        requests the cancellation and returns without waiting, as the query may be
        waiting for that thread: this includes Python code Polars runs on its
        threads, such as a Python function of a query, a scan resolver or the
        generator of a Python IO source.

        Dropping the query's handle (this object) also requests the query to stop,
        but doesn't wait for it: call `cancel()` to know it has stopped.

        As it waits for the query, it must not be called from another thread the
        query itself waits on: for example an asyncio event loop that a Python
        function of the query calls back into (with
        `asyncio.run_coroutine_threadsafe(...).result()`, say), which would never
        return. Call it off that thread instead, such as with
        `await loop.run_in_executor(None, query.cancel)`. A keyboard interrupt
        (Ctrl-C) while it waits raises `KeyboardInterrupt`; the query still stops.
        """
        self._inner.cancel()

    def fetch(self) -> DataFrame | None:
        """
        Fetch the result.

        If it is ready, a materialized DataFrame is returned.
        If it is not ready it will return `None`.
        """
        if (out := self._inner.fetch()) is not None:
            return wrap_df(out)
        else:
            return None

    def fetch_blocking(self) -> DataFrame:
        """Await the result synchronously."""
        return wrap_df(self._inner.fetch_blocking())
