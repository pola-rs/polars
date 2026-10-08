from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from polars.io.parquet.decryption import _check_bytes

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyFileEncryptionProperties

if TYPE_CHECKING:
    from collections.abc import Mapping


class ParquetEncryptionProperties:
    """
    Properties for writing Parquet files encrypted with Parquet modular encryption.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    footer_key
        The key used to encrypt the file footer, or to sign the footer if
        `plaintext_footer` is set. If no column keys are specified, all columns
        are also encrypted with the footer key.
    column_keys
        Keys used to encrypt columns with column-specific keys, keyed by column
        name. If any column keys are specified, only the columns with a key are
        encrypted and all other columns are written unencrypted. For nested
        columns, the column name is the dot-separated path in the Parquet schema,
        e.g. `a.b.c`.
    plaintext_footer
        Write the file footer in plaintext rather than encrypting it, so that
        readers without the footer key can read the schema and any unencrypted
        columns. The footer is still signed with the footer key.
    aad_prefix
        The AAD (additional authenticated data) prefix. This should uniquely
        identify the file, and protects against files being swapped or replaced.
    store_aad_prefix
        Store the AAD prefix in the file. If this is false, readers must provide
        the AAD prefix in order to read the file.

    Examples
    --------
    >>> encryption_properties = pl.ParquetEncryptionProperties(
    ...     footer_key=b"0123456789012345",
    ...     column_keys={"x": b"1234567890123450"},
    ... )
    """

    def __init__(
        self,
        *,
        footer_key: bytes,
        column_keys: Mapping[str, bytes] | None = None,
        plaintext_footer: bool = False,
        aad_prefix: bytes | None = None,
        store_aad_prefix: bool = False,
    ) -> None:
        _check_bytes(footer_key, "footer_key")
        column_keys_list = list(column_keys.items()) if column_keys is not None else []
        for column_name, key in column_keys_list:
            _check_bytes(key, f"key for column {column_name!r}")
        if aad_prefix is not None:
            _check_bytes(aad_prefix, "aad_prefix")
        elif store_aad_prefix:
            msg = "store_aad_prefix requires aad_prefix to be set"
            raise ValueError(msg)

        self._pyencryptionproperties = PyFileEncryptionProperties(
            footer_key,
            column_keys_list,
            plaintext_footer,
            aad_prefix,
            store_aad_prefix,
        )
