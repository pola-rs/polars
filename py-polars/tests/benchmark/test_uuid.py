"""Benchmarks for native UUID storage and operations."""

from __future__ import annotations

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()

N = 250_000


@pytest.fixture(scope="module")
def uuid_data() -> pl.DataFrame:
    native = pl.uuid4(N, eager=True).rename("native")
    return pl.DataFrame({"native": native, "string": native.cast(pl.String)})


def test_uuid_generate_v4() -> None:
    pl.uuid4(N, eager=True)


def test_uuid_generate_v7() -> None:
    pl.uuid7(N, eager=True)


def test_uuid_parse(uuid_data: pl.DataFrame) -> None:
    uuid_data.select(pl.col("string").cast(pl.UUID))


def test_uuid_format(uuid_data: pl.DataFrame) -> None:
    uuid_data.select(pl.col("native").cast(pl.String))


@pytest.mark.parametrize("column", ["native", "string"])
def test_uuid_sort(uuid_data: pl.DataFrame, column: str) -> None:
    uuid_data.select(pl.col(column).sort())


@pytest.mark.parametrize("column", ["native", "string"])
def test_uuid_n_unique(uuid_data: pl.DataFrame, column: str) -> None:
    uuid_data.select(pl.col(column).n_unique())


@pytest.mark.parametrize("column", ["native", "string"])
def test_uuid_equality_filter(uuid_data: pl.DataFrame, column: str) -> None:
    value = uuid_data[column][N // 2]
    uuid_data.filter(pl.col(column) == value)
