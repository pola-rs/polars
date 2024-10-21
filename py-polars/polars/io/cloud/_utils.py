from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polars._utils.various import is_path_or_str_sequence

if TYPE_CHECKING:
    from polars._typing import ScanSource


def _first_scan_path(
    source: ScanSource,
) -> str | Path | None:
    if isinstance(source, (str, Path)):
        return source
    elif is_path_or_str_sequence(source) and source:
        return source[0]

    return None


def _infer_cloud_type(
    source: ScanSource,
) -> Literal["aws", "azure", "gcp", "file", "http", "hf"] | None:
    if (path := _first_scan_path(source)) is None:
        return None

    scheme = _get_path_scheme(path)

    # Fast path - local file
    if not scheme:
        return None

    if scheme == "file":
        return "file"

    if _is_aws_cloud(scheme):
        return "aws"

    if _is_azure_cloud(scheme):
        return "azure"

    if _is_gcp_cloud(scheme):
        return "gcp"

    if _is_http_cloud(scheme):
        return "http"

    if _is_hf_cloud(scheme):
        return "hf"

    return None


def _get_path_scheme(path: str | Path) -> str | None:
    splitted = str(path).split("://", maxsplit=1)

    return None if not splitted else splitted[0]


def _is_aws_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["s3", "s3a"])


def _is_azure_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["az", "azure", "adl", "abfs", "abfss"])


def _is_gcp_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["gs", "gcp", "gcs"])


def _is_http_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["http", "https"])


def _is_hf_cloud(scheme: str) -> bool:
    return scheme == "hf"
