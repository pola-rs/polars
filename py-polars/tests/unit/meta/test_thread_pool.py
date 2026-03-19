from __future__ import annotations

import pytest

import polars as pl


def test_thread_pool_size() -> None:
    result = pl.thread_pool_size()
    assert isinstance(result, int)


def test_threadpool_size_deprecated() -> None:
    with pytest.deprecated_call():
        result = pl.threadpool_size()
    assert isinstance(result, int)
