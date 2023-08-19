from __future__ import annotations

from typing import TYPE_CHECKING

from polars.utils._wrap import wrap_df

if TYPE_CHECKING:
    from queue import Queue

    from polars import DataFrame


class _AsyncDataFrameResult:
    queue: Queue
    _result: DataFrame | Exception

    def __init__(self, queue: Queue[DataFrame | Exception]) -> None:
        self.queue = queue
        self._result = None

    def get(self, block=True, timeout=None) -> DataFrame:
        if self._result is not None:
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

        self._result = self.queue.get(block=block, timeout=timeout)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def _callback(self, obj):
        if not isinstance(obj, Exception):
            obj = wrap_df(obj)
        self.queue.put_nowait(obj)
