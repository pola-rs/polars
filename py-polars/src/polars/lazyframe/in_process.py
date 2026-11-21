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
        fetched, returns at once. Called from one of Polars' own threads (inside a
        Python function a query runs, say), it only requests the cancellation and
        returns without waiting, as the query may be waiting for that thread.
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
