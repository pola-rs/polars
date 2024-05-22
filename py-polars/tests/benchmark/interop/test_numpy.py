"""Benchmark tests for conversions from/to NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


@pytest.fixture(scope="module")
def floats_array() -> np.ndarray[Any, Any]:
    n_rows = 10_000
    return np.random.randn(n_rows)


@pytest.fixture()
def floats(floats_array: np.ndarray[Any, Any]) -> pl.Series:
    return pl.Series(floats_array)


@pytest.fixture()
def floats_with_nulls(floats: pl.Series) -> pl.Series:
    null_probability = 0.1
    validity = pl.Series(np.random.uniform(size=floats.len())) > null_probability
    return pl.select(pl.when(validity).then(floats)).to_series()


@pytest.fixture()
def floats_chunked(floats_array: np.ndarray[Any, Any]) -> pl.Series:
    n_chunks = 5
    chunk_len = len(floats_array) // n_chunks
    chunks = [
        floats_array[i * chunk_len : (i + 1) * chunk_len] for i in range(n_chunks)
    ]
    chunks_copy = [pl.Series(c.copy()) for c in chunks]
    return pl.concat(chunks_copy, rechunk=False)


def test_to_numpy_series_zero_copy(floats: pl.Series) -> None:
    floats.to_numpy()


def test_to_numpy_series_with_nulls(floats_with_nulls: pl.Series) -> None:
    floats_with_nulls.to_numpy()


def test_to_numpy_series_chunked(floats_chunked: pl.Series) -> None:
    floats_chunked.to_numpy()
