from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from polars.utils._wrap import wrap_df

if TYPE_CHECKING:
    from queue import Queue

    from polars.polars import PyDataFrame


T = TypeVar("T")


class _AsyncDataFrameResult(Generic[T]):
    queue: Queue[Exception | T]
    _result: Exception | T | None

    __slots__ = ("queue", "_result")

    def __init__(self, queue: Queue[Exception | T]) -> None:
        self.queue = queue
        self._result = None

    def get(self, **kwargs: Any) -> T:
        if self._result is not None:
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

        self._result = self.queue.get(**kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def _callback(self, obj: PyDataFrame | Exception) -> None:
        if not isinstance(obj, Exception):
            obj = wrap_df(obj)
        self.queue.put_nowait(obj)

    def _callback_all(self, obj: list[PyDataFrame] | Exception) -> None:
        if not isinstance(obj, Exception):
            obj = [wrap_df(pydf) for pydf in obj]
        self.queue.put_nowait(obj)  # type: ignore[arg-type]
