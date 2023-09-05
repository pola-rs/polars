from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import pytest
from moto.server import ThreadedMotoServer

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.xdist_group("aws"),
    pytest.mark.slow(),
]

port = 5000
host = "127.0.0.1"
region = "us-east-1"
files = ["foods1.csv", "foods1.ipc", "foods1.parquet"]


@pytest.fixture(scope="session")
def s3_base() -> Iterator[str]:
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "accesskey"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "secretkey"
    moto_server = ThreadedMotoServer(host, port)
    moto_server.start()
    print("server up")
    yield f"http://{host}:{port}"
    print("moto done")
    moto_server.stop()


@pytest.fixture()
def s3(s3_base: str, io_files_path: Path) -> str:
    import boto3

    client = boto3.client("s3", region_name=region, endpoint_url=s3_base)
    client.create_bucket(Bucket="bucket")
    for file in files:
        client.upload_file(io_files_path / file, Bucket="bucket", Key=file)
    return s3_base


def test_read_csv_on_s3(s3: str) -> None:
    df = pl.read_csv(
        "s3://bucket/foods1.csv",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (27, 4)


def test_read_ipc_on_s3(s3: str) -> None:
    df = pl.read_ipc(
        "s3://bucket/foods1.ipc",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (27, 4)


def test_scan_ipc_on_s3(s3: str) -> None:
    df = pl.scan_ipc(
        "s3://bucket/foods1.ipc",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.collect().shape == (27, 4)


def test_read_parquet_on_s3(s3: str) -> None:
    df = pl.read_parquet(
        "s3://bucket/foods1.parquet",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (27, 4)


def test_scan_parquet_on_s3(s3: str) -> None:
    df = pl.scan_parquet(
        "s3://bucket/foods1.parquet",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.collect().shape == (27, 4)
