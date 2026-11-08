from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


# Binds a function with a thread pool.
# Used from Rust, to allow the following:
#   (py, func, args, kwargs, pool)
#     -> FnPoolWrap.call0(py, func, pool).call(py, args, kwargs)
class FnPoolWrap(Generic[P, T]):
    def __init__(self, f: Callable[P, T], pool_wrap: PyThreadPool) -> None:
        self.f = f
        self.pool_wrap = pool_wrap

    def __call__(self, *a: P.args, **kw: P.kwargs) -> T:
        try:
            return self.pool_wrap.pool.submit(self.f, *a, **kw).result()
        except BaseException as e:
            if self.pool_wrap.last_exception is None:
                self.pool_wrap.last_exception = e

            # Shutdown, otherwise exception doesn't get raised until all tasks
            # finish.
            self.pool_wrap.pool.shutdown(wait=False, cancel_futures=True)

            raise self.pool_wrap.last_exception from e


class PyThreadPool:
    def __init__(self, num_threads: int) -> None:
        self.pool = ThreadPoolExecutor(num_threads)
        self.last_exception: BaseException | None = None
