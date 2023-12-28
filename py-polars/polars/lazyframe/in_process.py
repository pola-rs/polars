from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_df

if TYPE_CHECKING:
    from polars import DataFrame
    from polars.polars import PyInProcessQuery


class InProcessQuery:
    """
    A placeholder for an in process query.

    This can be used to do something else while a query is running.
    The queries can be cancelled. You can peek if the query is finished,
    or you can await the result.
    """

    def __init__(self, ipq: PyInProcessQuery) -> None:
        self.ipq = ipq

    def cancel(self) -> None:
        """Cancel the query at earliest convenience."""
        self.ipq.cancel()

    def fetch(self) -> DataFrame | None:
        """
        Fetch the result.

        If it is ready, a materialized DataFrame is returned.
        If it is not ready it will return `None`.
        """
        out = self.ipq.fetch()
        if out is not None:
            return wrap_df(out)
        return None

    def fetch_blocking(self) -> DataFrame:
        """Await the result synchronously."""
        return wrap_df(self.ipq.fetch_blocking())
