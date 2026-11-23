from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Literal

from polars.io.cloud._utils import NoPickleOption
from polars.io.lance._scan_resolver import LanceScanResolver

if TYPE_CHECKING:
    import lance
    import lancedb
    from lance import LanceDataset

    import polars as pl
    from polars._typing import StorageOptionsDict
    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderFunction,
    )


def scan_lance(
    source: str | lance.LanceDataset | lancedb.Table,
    *,
    version: int | str | None = None,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
) -> pl.LazyFrame:
    """
    Lazily read from a Lance dataset.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    .. engine-support:: in-memory, streaming, distributed

    Parameters
    ----------
    source
        Path to the lance dataset, or a LanceDataset object.
    version
        If specified, load a specific version of the Lance dataset. Else, loads the
        latest version. A version number (`int`) or a tag (`str`) can be provided.
    storage_options
        Extra options for the storage backends supported by `pylance`.
        For cloud storages, this may include configurations for authentication etc.
    credential_provider
        Provide a function that can be called to provide cloud storage
        credentials. The function is expected to return a dictionary of
        credential keys along with an optional credential expiry time.
    """
    from polars.io.cloud.credential_provider._builder import (
        _init_credential_provider_builder,
    )

    dataset: LanceDataset | None = None

    if importlib.util.find_spec("lance") is not None:
        import lance

        if isinstance(source, lance.LanceDataset):
            dataset = source

        elif importlib.util.find_spec("lancedb") is not None:
            import lancedb

            if isinstance(source, lancedb.Table):
                dataset = source.to_lance()

    if (
        dataset is not None
        and (accessor := dataset.storage_options_accessor) is not None
    ):
        storage_options = {
            **(accessor.get_storage_options()),
            **(storage_options or {}),
        }

    if dataset is None:
        credential_provider_builder = _init_credential_provider_builder(
            credential_provider, source, storage_options, "scan_lance"
        )
    elif credential_provider is not None and credential_provider != "auto":
        msg = "cannot use credential_provider when passing a LanceDataset object"
        raise ValueError(msg)
    else:
        credential_provider_builder = None

    del credential_provider

    return LanceScanResolver(
        dataset_=NoPickleOption(dataset),
        dataset_uri_=str(source) if dataset is None else None,
        version=version,
        storage_options=storage_options,
        credential_provider_builder=credential_provider_builder,
    ).lazy()
