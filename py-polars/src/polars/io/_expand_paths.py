from __future__ import annotations

import contextlib
from typing import IO, TYPE_CHECKING, Literal

from polars._utils.wrap import wrap_ldf
from polars.io._utils import get_sources
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)
from polars.io.scan_options._options import ScanOptions

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyLazyFrame

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polars._typing import StorageOptionsDict
    from polars.io.cloud import CredentialProviderFunction
    from polars.lazyframe.frame import LazyFrame


def _expand_paths(
    source: (
        str
        | Path
        | IO[str]
        | IO[bytes]
        | bytes
        | list[str]
        | list[Path]
        | list[IO[str]]
        | list[IO[bytes]]
    ),
    *,
    glob: bool = True,
    hidden_file_prefix: str | Sequence[str] | None = None,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
) -> LazyFrame:
    sources = get_sources(source)

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, sources, storage_options, "expand_paths"
    )
    del credential_provider

    pylf = PyLazyFrame.new_from_expand_paths(
        sources=sources,
        scan_options=ScanOptions(
            row_index=None,
            pre_slice=None,
            include_file_paths=None,
            glob=glob,
            hidden_file_prefix=(
                [hidden_file_prefix]
                if isinstance(hidden_file_prefix, str)
                else hidden_file_prefix
            ),
            storage_options=storage_options,
            credential_provider=credential_provider_builder,
        ),
        name="path",
    )

    return wrap_ldf(pylf)
