"""Moto-backed S3 server shared by the IO tests."""

from __future__ import annotations

import threading
from collections import Counter
from typing import TYPE_CHECKING, Any

import boto3
from moto.server import (  # type: ignore[attr-defined]
    DomainDispatcherApplication,
    create_backend_app,
)
from werkzeug.serving import make_server

if TYPE_CHECKING:
    from collections.abc import Callable


ACCESS_KEY = "accesskey"
SECRET_KEY = "secretkey"
REGION = "us-east-1"


class CountingS3:
    """Moto S3 server with request counts and an arrival-ordered log.

    `requests` counts by method and range; `log` records (method, bucket/key, range).
    """

    def __init__(self) -> None:
        app = DomainDispatcherApplication(create_backend_app)
        self.requests: Counter[str] = Counter()
        self.log: list[tuple[str, str, str]] = []
        # Run before moto; any non-None return value triggers an empty HTTP 500.
        self.on_request: Callable[[dict[str, Any]], Any] | None = None
        lock = threading.Lock()

        def counting_app(environ: dict[str, Any], start_response: Any) -> Any:
            method = environ["REQUEST_METHOD"]
            range_header = environ.get("HTTP_RANGE", "")
            with lock:
                self.requests[f"{method} {range_header}"] += 1
                self.log.append(
                    (method, environ.get("PATH_INFO", "").lstrip("/"), range_header)
                )
            if self.on_request is not None:
                response = self.on_request(environ)
                if response is not None:
                    start_response("500 Internal Server Error", [])
                    return [b""]
            return app(environ, start_response)

        self.server = make_server("127.0.0.1", 0, counting_app, threaded=True)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
        )

    @property
    def storage_options(self) -> dict[str, str]:
        return {
            "aws_access_key_id": ACCESS_KEY,
            "aws_secret_access_key": SECRET_KEY,
            "aws_region": REGION,
            "aws_endpoint_url": self.endpoint,
            "aws_allow_http": "true",
        }

    def n_requests(self, prefix: str) -> int:
        return sum(v for k, v in self.requests.items() if k.startswith(prefix))

    def range_get_keys(self, log: list[tuple[str, str, str]]) -> set[str]:
        """Return keys fetched by range GET, counting multi-request footers once."""
        return {key for method, key, rng in log if method == "GET" and rng}

    def since(self, mark: int) -> list[tuple[str, str, str]]:
        return self.log[mark:]
