from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.cloud.conftest import SECRET_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from tests.unit.io.cloud.conftest import CountingS3

pytest.importorskip("pyiceberg")


@pytest.mark.write_disk
def test_iceberg_catalog_storage_options(
    s3: CountingS3,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    from pyiceberg.catalog.sql import SqlCatalog

    # only the catalog gives credentials: not the environment, nor a profile
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    for var in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent"))
    # Polars reads the data files itself, planned by PyIceberg
    monkeypatch.setenv("POLARS_ICEBERG_PLANNER", "pyiceberg")

    catalog = SqlCatalog(
        "c",
        uri=f"sqlite:///{tmp_path / 'c.db'}",
        warehouse=f"file://{tmp_path}",
        **{
            "s3.endpoint": s3.endpoint,
            "s3.access-key-id": "catalog-key",
            "s3.secret-access-key": SECRET_KEY,
            "s3.session-token": "catalog-token",
            "s3.region": "eu-west-1",
        },
    )
    request.addfinalizer(catalog.engine.dispose)
    catalog.create_namespace("ns")
    df = pl.DataFrame({"a": [1, 2]})
    # data files on S3; metadata on disk, which PyIceberg writes
    table = catalog.create_table(
        ("ns", "t"),
        schema=df.to_arrow().schema,
        location="s3://bucket/t",
        properties={"write.metadata.path": f"file://{tmp_path / 'metadata'}"},
    )
    # moto doesn't check signatures: check which credentials signed each request
    signers: set[tuple[str, str]] = set()

    def record(environ: dict[str, Any]) -> None:
        auth = environ.get("HTTP_AUTHORIZATION", "")
        credential = auth.partition("Credential=")[2].split(",")[0].split("/")
        signers.add(
            (
                f"{credential[0]}/{credential[2] if len(credential) > 2 else ''}",
                environ.get("HTTP_X_AMZ_SECURITY_TOKEN", ""),
            )
        )

    s3.on_request = record
    df.lazy().sink_iceberg(table, mode="append")
    df.lazy().sink_iceberg(catalog.load_table(("ns", "t")), mode="append")
    out = pl.scan_iceberg(catalog.load_table(("ns", "t"))).collect()
    assert_frame_equal(out, pl.concat([df, df]))
    assert any(path.startswith("bucket/t/") for _, path, _ in s3.log)
    assert signers == {("catalog-key/eu-west-1", "catalog-token")}

    # the user's credentials replace the catalog's, its region and endpoint stay
    signers.clear()
    user = {"aws_access_key_id": "user-key", "aws_secret_access_key": SECRET_KEY}
    df.lazy().sink_iceberg(
        catalog.load_table(("ns", "t")), mode="append", storage_options=user
    )
    lf = pl.scan_iceberg(catalog.load_table(("ns", "t")), storage_options=user)
    assert_frame_equal(lf.collect(), pl.concat([df, df, df]))
    assert signers == {("user-key/eu-west-1", "")}
