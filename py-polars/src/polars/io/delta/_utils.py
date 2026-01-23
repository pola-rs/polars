from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polars._dependencies import _DELTALAKE_AVAILABLE, deltalake
from polars.datatypes import Null, Time
from polars.datatypes.convert import unpack_dtypes
from polars.io.cloud._utils import POLARS_STORAGE_CONFIG_KEYS, _get_path_scheme

if TYPE_CHECKING:
    from deltalake import DeltaTable

    from polars import DataType
    from polars._typing import StorageOptionsDict


def _resolve_delta_lake_uri(table_uri: str | Path, *, strict: bool = True) -> str:
    resolved_uri = str(
        Path(table_uri).expanduser().resolve(strict)
        if _get_path_scheme(table_uri) is None
        else table_uri
    )

    return resolved_uri


def _get_delta_lake_table(
    table_path: str | Path | DeltaTable,
    version: int | str | datetime | None = None,
    storage_options: StorageOptionsDict | None = None,
    delta_table_options: dict[str, Any] | None = None,
) -> deltalake.DeltaTable:
    """
    Initialize a Delta lake table for use in read and scan operations.

    Notes
    -----
    Make sure to install deltalake>=0.8.0. Read the documentation
    `here <https://delta-io.github.io/delta-rs/usage/installation/>`_.
    """
    _check_if_delta_available()

    if storage_options is not None:
        # Don't pass these to delta as it errors on non-string type values.
        storage_options = {
            k: v
            for k, v in storage_options.items()
            if k not in POLARS_STORAGE_CONFIG_KEYS
        }

    if isinstance(table_path, deltalake.DeltaTable):
        if any(
            [
                version is not None,
                storage_options is not None,
                delta_table_options is not None,
            ]
        ):
            warnings.warn(
                """When supplying a DeltaTable directly, `version`, `storage_options`, and `delta_table_options` are ignored.
                To silence this warning, don't supply those parameters.""",
                RuntimeWarning,
                stacklevel=1,
            )
        return table_path
    if delta_table_options is None:
        delta_table_options = {}
    resolved_uri = _resolve_delta_lake_uri(table_path)
    if not isinstance(version, (str, datetime)):
        dl_tbl = deltalake.DeltaTable(
            resolved_uri,
            version=version,
            storage_options=storage_options,
            **delta_table_options,
        )
    else:
        dl_tbl = deltalake.DeltaTable(
            table_path,
            storage_options=storage_options,
            **delta_table_options,
        )
        dl_tbl.load_as_version(version)

    return dl_tbl


def _check_if_delta_available() -> None:
    if not _DELTALAKE_AVAILABLE:
        msg = "deltalake is not installed\n\nPlease run: pip install deltalake"
        raise ModuleNotFoundError(msg)


def _check_for_unsupported_types(dtypes: list[DataType]) -> None:
    schema_dtypes = unpack_dtypes(*dtypes)
    unsupported_types = {Time, Null}
    # Note that this overlap check does NOT work correctly for Categorical, so
    # if Categorical is added back to unsupported_types a different check will
    # need to be used.

    if overlap := schema_dtypes & unsupported_types:
        msg = f"dataframe contains unsupported data types: {overlap!r}"
        raise TypeError(msg)
