from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_df

if TYPE_CHECKING:
    from queue import Queue

    from polars import DataFrame
    from polars.polars import PyDataFrame


class _AsyncDataFrameResult:
    queue: Queue[DataFrame | Exception]
    _result: DataFrame | Exception | None

    def __init__(self, queue: Queue[DataFrame | Exception]) -> None:
        self.queue = queue
        self._result = None

    def get(self, block: bool = True, timeout: float | None = None) -> DataFrame:
        if self._result is not None:
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

        self._result = self.queue.get(block=block, timeout=timeout)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def _callback(self, obj: PyDataFrame | Exception) -> None:
        if not isinstance(obj, Exception):
            obj = wrap_df(obj)
        self.queue.put_nowait(obj)
