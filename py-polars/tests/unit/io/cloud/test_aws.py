from __future__ import annotations

from typing import TYPE_CHECKING, Any

import boto3
import pytest
from moto.server import ThreadedMotoServer

import polars as pl
from polars.testing import assert_frame_equal
from tests.conftest import PlMonkeyPatch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

pytestmark = [
    pytest.mark.xdist_group("aws"),
    pytest.mark.slow(),
]


@pytest.fixture(scope="module")
def monkeypatch_module() -> Any:
    """Allow module-scoped monkeypatching."""
    with PlMonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def s3_base(monkeypatch_module: Any) -> Iterator[str]:
    host = "127.0.0.1"
    port = 5000
    endpoint_uri = f"http://{host}:{port}"
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "accesskey")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "secretkey")
    monkeypatch_module.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch_module.setenv("AWS_ENDPOINT_URL", endpoint_uri)

    moto_server = ThreadedMotoServer(host, port)
    moto_server.start()
    print("server up")
    yield endpoint_uri
    print("moto done")
    moto_server.stop()


@pytest.fixture
def s3(s3_base: str, io_files_path: Path) -> str:
    region = "us-east-1"
    client = boto3.client("s3", region_name=region, endpoint_url=s3_base)
    client.create_bucket(Bucket="bucket")

    files = [
        "foods1.csv",
        "foods2.csv",
        "foods1.ipc",
        "foods1.parquet",
        "foods2.parquet",
    ]
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
    storage_options = {"endpoint_url": s3}
    df = function(
        f"s3://bucket/foods1.{extension}",
        storage_options=storage_options,
    )
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (27, 4)

    # ensure we aren't modifying the original user dictionary (ref #15859)
    assert storage_options == {"endpoint_url": s3}


@pytest.mark.parametrize(
    ("function", "extension"),
    [
        (pl.scan_csv, "csv"),
        (pl.scan_ipc, "ipc"),
        (pl.scan_parquet, "parquet"),
    ],
)
def test_scan_s3(s3: str, function: Callable[..., Any], extension: str) -> None:
    lf = function(
        f"s3://bucket/foods1.{extension}",
        storage_options={"endpoint_url": s3},
    )
    assert lf.collect_schema().names() == ["category", "calories", "fats_g", "sugars_g"]
    assert lf.collect().shape == (27, 4)


def test_lazy_count_s3(s3: str) -> None:
    lf = pl.scan_csv(
        "s3://bucket/foods*.csv", storage_options={"endpoint_url": s3}
    ).select(pl.len())

    assert "FAST COUNT" in lf.explain()
    expected = pl.DataFrame({"len": [54]}, schema={"len": pl.UInt32})
    assert_frame_equal(lf.collect(), expected)


def test_read_parquet_metadata(s3: str) -> None:
    metadata = pl.read_parquet_metadata(
        "s3://bucket/foods1.parquet", storage_options={"endpoint_url": s3}
    )
    assert "ARROW:schema" in metadata
