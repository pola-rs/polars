from __future__ import annotations

import multiprocessing
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
    pytest.mark.skip(
        reason="Causes intermittent failures in CI. See: "
        "https://github.com/pola-rs/polars/issues/16910"
    ),
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
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "accesskey")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "secretkey")
    monkeypatch_module.setenv("AWS_DEFAULT_REGION", "us-east-1")

    host = "127.0.0.1"
    port = 5000
    moto_server = ThreadedMotoServer(host, port)
    # Start in a separate process to avoid deadlocks
    mp = multiprocessing.get_context("spawn")
    p = mp.Process(target=moto_server._server_entry, daemon=True)
    p.start()
    print("server up")
    yield f"http://{host}:{port}"
    print("moto done")
    p.kill()


@pytest.fixture
def s3(s3_base: str, io_files_path: Path) -> str:
    region = "us-east-1"
    client = boto3.client("s3", region_name=region, endpoint_url=s3_base)
    client.create_bucket(Bucket="bucket")

    files = ["foods1.csv", "foods1.ipc", "foods1.parquet", "foods2.parquet"]
    for file in files:
        client.upload_file(io_files_path / file, Bucket="bucket", Key=file)
    return s3_base


@pytest.mark.parametrize(
    ("function", "extension"),
    [
        (pl.read_csv, "csv"),
        (pl.read_ipc, "ipc"),
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
    [(pl.scan_ipc, "ipc"), (pl.scan_parquet, "parquet")],
)
def test_scan_s3(s3: str, function: Callable[..., Any], extension: str) -> None:
    lf = function(
        f"s3://bucket/foods1.{extension}",
        storage_options={"endpoint_url": s3},
    )
    assert lf.collect_schema().names() == ["category", "calories", "fats_g", "sugars_g"]
    assert lf.collect().shape == (27, 4)


def test_lazy_count_s3(s3: str) -> None:
    lf = pl.scan_parquet(
        "s3://bucket/foods*.parquet", storage_options={"endpoint_url": s3}
    ).select(pl.len())

    assert "FAST_COUNT" in lf.explain()
    expected = pl.DataFrame({"len": [54]}, schema={"len": pl.UInt32})
    assert_frame_equal(lf.collect(), expected)


def test_read_parquet_metadata(s3: str) -> None:
    metadata = pl.read_parquet_metadata(
        "s3://bucket/foods1.parquet", storage_options={"endpoint_url": s3}
    )
    assert "ARROW:schema" in metadata


def test_scan_parquet_file_statistics(
    s3: str,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    storage_options = {"endpoint_url": s3}
    client = boto3.client("s3", region_name="us-east-1", endpoint_url=s3)

    # Use 1000 columns to create large-enough metadata to run into re-fetches
    df = pl.DataFrame({f"a{i}": ["a", "b", "c"] for i in range(1000)})
    for i in range(2):
        local_path = tmp_path / f"test_{i}.parquet"
        df.write_parquet(local_path)
        client.upload_file(str(local_path), Bucket="bucket", Key=f"test_{i}.parquet")

    # Read footer size from the local copy
    local_path = tmp_path / "test_1.parquet"
    file_size = local_path.stat().st_size
    with local_path.open("rb") as f:
        f.seek(-8, 2)
        thrift_metadata_size = int.from_bytes(f.read(4), "little")

    expected = pl.concat([df, df])

    # Without file_statistics, an extra fetch is needed for the second file
    capfd.readouterr()
    result = pl.scan_parquet(
        "s3://bucket/test_*.parquet",
        storage_options=storage_options,
    ).collect()
    assert_frame_equal(result, expected)
    stderr = capfd.readouterr().err
    assert "bytes need to be fetched" in stderr

    # With the correct footer size hint for file index 1, no extra fetch
    capfd.readouterr()
    result = pl.scan_parquet(
        "s3://bucket/test_*.parquet",
        storage_options=storage_options,
        _file_statistics={1: (file_size, thrift_metadata_size)},
    ).collect()
    assert_frame_equal(result, expected)
    stderr = capfd.readouterr().err
    assert "bytes need to be fetched" not in stderr
    assert "Fetched all bytes for metadata on first try" in stderr
