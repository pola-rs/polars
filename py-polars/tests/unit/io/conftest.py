from __future__ import annotations

import os
from pathlib import Path

import pytest

import polars as pl

IO_TEST_DIR = os.path.abspath(os.path.dirname(__file__))

EXAMPLES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "examples",
        "datasets",
    )
)

FOODS_CSV = os.path.join(
    EXAMPLES_DIR,
    "foods1.csv",
)


FOODS_CSV_GLOB = os.path.join(
    EXAMPLES_DIR,
    "foods*.csv",
)

FOODS_PARQUET = os.path.join(
    EXAMPLES_DIR,
    "foods1.parquet",
)

FOODS_IPC = os.path.join(
    EXAMPLES_DIR,
    "foods1.ipc",
)

FOODS_NDJSON = os.path.join(
    EXAMPLES_DIR,
    "foods1.ndjson",
)


@pytest.fixture()
def io_test_dir() -> str:
    return IO_TEST_DIR


@pytest.fixture()
def io_files_path() -> Path:
    return Path(IO_TEST_DIR) / "files"


@pytest.fixture()
def examples_dir() -> str:
    return EXAMPLES_DIR


@pytest.fixture()
def foods_csv() -> str:
    return FOODS_CSV


@pytest.fixture()
def foods_csv_glob() -> str:
    return FOODS_CSV


if not os.path.isfile(FOODS_PARQUET):
    pl.read_csv(FOODS_CSV).write_parquet(FOODS_PARQUET)

if not os.path.isfile(FOODS_IPC):
    pl.read_csv(FOODS_CSV).write_ipc(FOODS_IPC)

if not os.path.isfile(FOODS_NDJSON):
    pl.read_csv(FOODS_CSV).write_ndjson(FOODS_NDJSON)


@pytest.fixture()
def foods_ipc() -> str:
    return FOODS_IPC


@pytest.fixture()
def foods_parquet() -> str:
    return FOODS_PARQUET


@pytest.fixture()
def foods_ndjson() -> str:
    return FOODS_NDJSON
