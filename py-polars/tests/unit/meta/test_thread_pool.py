from __future__ import annotations

import polars as pl


def test_thread_pool_size() -> None:
    result = pl.thread_pool_size()
    assert isinstance(result, int)
