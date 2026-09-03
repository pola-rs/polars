from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any


# Binds a function with a thread pool.
# Used from Rust, to allow the following:
#   (py, func, args, kwargs, pool)
#     -> FnPoolWrap.call0(py, func, pool).call(py, args, kwargs)
class FnPoolWrap:
    def __init__(self, f: Any, pool_wrap: PyScanResolveThreadPool) -> None:
        self.f = f
        self.pool_wrap = pool_wrap

    def __call__(self, *a: Any, **kw: Any) -> Any:
        try:
            return self.pool_wrap.pool.submit(self.f, *a, **kw).result()
        except BaseException as e:
            if self.pool_wrap.last_exception is None:
                self.pool_wrap.last_exception = e

            # Shutdown, otherwise exception doesn't get raised until all tasks
            # finish.
            self.pool_wrap.pool.shutdown(wait=False, cancel_futures=True)

            raise self.pool_wrap.last_exception from e


class PyScanResolveThreadPool:
    def __init__(self, num_threads: int) -> None:
        self.pool = ThreadPoolExecutor(num_threads)
        self.last_exception: Any = None
