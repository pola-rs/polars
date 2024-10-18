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

    splitted = str(path).split("://", maxsplit=1)

    # Fast path - local file
    if not splitted:
        return None

    scheme = splitted[0]

    if scheme == "file":
        return "file"

    if any(scheme == x for x in ["s3", "s3a"]):
        return "aws"

    if any(scheme == x for x in ["az", "azure", "adl", "abfs", "abfss"]):
        return "azure"

    if any(scheme == x for x in ["gs", "gcp", "gcs"]):
        return "gcp"

    if any(scheme == x for x in ["http", "https"]):
        return "http"

    if scheme == "hf":
        return "hf"

    return None
