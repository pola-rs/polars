"""Fixtures for the cloud IO tests."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from tests.unit.io.cloud.moto_server import CountingS3

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def s3() -> Iterator[CountingS3]:
    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    s3.client.create_bucket(Bucket="bucket")
    yield s3
    s3.server.shutdown()
