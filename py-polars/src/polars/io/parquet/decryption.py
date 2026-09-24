from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from polars._utils.various import qualified_type_name

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyFileDecryptionProperties

if TYPE_CHECKING:
    from collections.abc import Mapping


class ParquetDecryptionProperties:
    """
    Properties for reading Parquet files encrypted with Parquet modular encryption.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    footer_key
        The key used to decrypt the file footer. This is also used to decrypt
        column data for files where all columns are encrypted with the footer key.
    column_keys
        Keys used to decrypt columns that are encrypted with column-specific keys,
        keyed by column name. For nested columns, the column name is the
        dot-separated path in the Parquet schema, e.g. `a.b.c`.
    aad_prefix
        The AAD (additional authenticated data) prefix. This must be provided if
        the file was written with an AAD prefix that is not stored in the file.
    verify_footer_signature
        Verify the footer signature of encrypted files with a plaintext footer.

    Examples
    --------
    >>> decryption_properties = pl.ParquetDecryptionProperties(
    ...     footer_key=b"0123456789012345",
    ...     column_keys={"x": b"1234567890123450"},
    ... )
    """

    def __init__(
        self,
        *,
        footer_key: bytes,
        column_keys: Mapping[str, bytes] | None = None,
        aad_prefix: bytes | None = None,
        verify_footer_signature: bool = True,
    ) -> None:
        _check_bytes(footer_key, "footer_key")
        column_keys_list = list(column_keys.items()) if column_keys is not None else []
        for column_name, key in column_keys_list:
            _check_bytes(key, f"key for column {column_name!r}")
        if aad_prefix is not None:
            _check_bytes(aad_prefix, "aad_prefix")

        self._pydecryptionproperties = PyFileDecryptionProperties(
            footer_key,
            column_keys_list,
            aad_prefix,
            verify_footer_signature,
        )


def _check_bytes(value: Any, name: str) -> None:
    if not isinstance(value, bytes):
        msg = f"{name} must be bytes, got {qualified_type_name(value)!r}"
        raise TypeError(msg)
