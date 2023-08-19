from queue import Queue

from polars import DataFrame
from polars.utils._wrap import wrap_df


class _AsyncDataFrameResult:
    queue: Queue
    _result: DataFrame

    def __init__(self, queue) -> None:
        self.queue = queue
        self._result = None

    def get(self, block=True, timeout=None) -> DataFrame:
        if self._result is not None:
            return self._result
        self._result = self.queue.get(block=block, timeout=timeout)
        return self._result

    def _callback(self, df):
        self.queue.put_nowait(wrap_df(df))
