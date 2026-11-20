from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit
from uuid import UUID

import polars as pl
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from polars._typing import DeletionFiles

_Z85 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"


def _decode_z85(value: str) -> bytes:
    if len(value) % 5:
        msg = "invalid Delta deletion vector Z85 length"
        raise ComputeError(msg)
    if len(value) // 5 * 4 > 512 * 1024 * 1024:
        msg = "deletion vector exceeds 512 MiB serialized size resource limit"
        raise ComputeError(msg)
    out = bytearray()
    try:
        for offset in range(0, len(value), 5):
            n = 0
            for c in value[offset : offset + 5]:
                n = n * 85 + _Z85.index(c)
            out.extend(n.to_bytes(4, "big"))
    except (ValueError, OverflowError) as e:
        msg = "invalid Delta deletion vector Z85 encoding"
        raise ComputeError(msg) from e
    return bytes(out)


def _native_deletion_vectors(
    add_actions: pl.DataFrame, paths: list[str], table_root: str
) -> DeletionFiles | None:
    """Use descriptor-capable delta-rs; None requests the older callback fallback."""
    if "deletion_vector" not in add_actions:
        return None
    expected = {
        "storageType": pl.String,
        "pathOrInlineDv": pl.String,
        "offset": pl.Int32,
        "sizeInBytes": pl.Int32,
        "cardinality": pl.Int64,
    }
    dtype = add_actions.schema["deletion_vector"]
    if not isinstance(dtype, pl.Struct) or dict(dtype.to_schema()) != expected:
        msg = f"invalid Delta deletion_vector descriptor schema: {dtype}"
        raise ComputeError(msg)

    # Normalize local URI schemes before applying delta-rs's file_uris escaping:
    # file:/// paths are local paths and must be decoded, rather than URL-quoted.
    from polars.io.delta._scan_resolver import _dv_path_key

    def path_key(path: str) -> str:
        path = path.removeprefix("file://").replace("lakefs://", "s3://", 1)
        if len(path) > 2 and path[0] == "/" and path[2] == ":" and path[1].isalpha():
            path = path[1:]
        return _dv_path_key(path, table_root)

    indices = {path: i for i, path in enumerate(paths)}
    out: dict[int, tuple[str | bytes, int, int, int]] = {}
    for path, dv in add_actions.select("path", "deletion_vector").iter_rows():
        if dv is None:
            continue
        source_path = path_key(
            path if urlsplit(path).scheme or path.startswith("/") else table_root + path
        )
        if source_path not in indices:
            msg = f"cannot match Delta deletion vector data file: {path}"
            raise ComputeError(msg)
        storage, location = dv["storageType"], dv["pathOrInlineDv"]
        size, cardinality, offset = dv["sizeInBytes"], dv["cardinality"], dv["offset"]
        if (
            storage not in ("i", "u", "p")
            or not isinstance(location, str)
            or size is None
            or size < 0
            or cardinality is None
            or cardinality < 0
            or (offset is not None and offset < 0)
        ):
            msg = "invalid Delta deletion vector descriptor"
            raise ComputeError(msg)
        if size > 512 * 1024 * 1024:
            msg = "deletion vector exceeds 512 MiB serialized size resource limit"
            raise ComputeError(msg)
        if storage == "i":
            if offset is not None:
                msg = "inline Delta deletion vector must not have an offset"
                raise ComputeError(msg)
            decoded = _decode_z85(location)
            if len(decoded) < size or len(decoded) - size > 3 or any(decoded[size:]):
                msg = "inline Delta deletion vector size mismatch"
                raise ComputeError(msg)
            value: str | bytes = decoded[:size]
        elif storage == "u":
            if len(location) < 20:
                msg = "invalid Delta deletion vector UUID"
                raise ComputeError(msg)
            prefix, encoded = location[:-20], location[-20:]
            decoded_prefix = unquote(prefix)
            # Prefix is a table-relative directory, never an absolute path/URL.
            if (
                decoded_prefix.startswith("/")
                or ".." in decoded_prefix.split("/")
                or urlsplit(decoded_prefix).scheme
            ):
                msg = "invalid Delta deletion vector path prefix"
                raise ComputeError(msg)
            value = path_key(
                table_root
                + prefix
                + ("/" if prefix else "")
                + f"deletion_vector_{UUID(bytes=_decode_z85(encoded))}.bin"
            )
        else:
            if not urlsplit(location).scheme and not location.startswith("/"):
                msg = "Delta deletion vector p path must be absolute"
                raise ComputeError(msg)
            value = path_key(location)
        out[indices[source_path]] = (value, offset or 0, size, cardinality)
    return "delta-deletion-vector-native", out
