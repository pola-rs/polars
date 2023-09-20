from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator

import boto3
import pytest
from moto.server import ThreadedMotoServer

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.xdist_group("aws"),
    pytest.mark.slow(),
]


@pytest.fixture(scope="module")
def monkeypatch_module() -> Any:
    """Allow module-scoped monkeypatching."""
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def s3_base(monkeypatch_module: Any) -> Iterator[str]:
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "accesskey")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "secretkey")
    monkeypatch_module.setenv("AWS_DEFAULT_REGION", "us-east-1")

    host = "127.0.0.1"
    port = 5000
    moto_server = ThreadedMotoServer(host, port)

    moto_server.start()
    print("server up")
    yield f"http://{host}:{port}"
    print("moto done")
    moto_server.stop()


@pytest.fixture()
def s3(s3_base: str, io_files_path: Path) -> str:
    region = "us-east-1"
    client = boto3.client("s3", region_name=region, endpoint_url=s3_base)
    client.create_bucket(Bucket="bucket")

    files = ["foods1.csv", "foods1.ipc", "foods1.parquet"]
    for file in files:
        client.upload_file(io_files_path / file, Bucket="bucket", Key=file)
    return s3_base


@pytest.mark.parametrize(
    ("function", "extension"),
    [
        (pl.read_csv, "csv"),
        (pl.read_ipc, "ipc"),
        (pl.read_parquet, "parquet"),
    ],
)
def test_read_s3(s3: str, function: Callable[..., Any], extension: str) -> None:
    df = function(
        f"s3://bucket/foods1.{extension}",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (27, 4)


@pytest.mark.parametrize(
    ("function", "extension"),
    [
        (pl.scan_ipc, "ipc"),
    ],
)
def test_scan_s3(s3: str, function: Callable[..., Any], extension: str) -> None:
    df = function(
        f"s3://bucket/foods1.{extension}",
        storage_options={"endpoint_url": s3},
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.collect().shape == (27, 4)
