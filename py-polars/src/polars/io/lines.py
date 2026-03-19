from __future__ import annotations

import contextlib
from typing import IO, TYPE_CHECKING, Literal

from polars._utils.unstable import unstable
from polars._utils.wrap import wrap_ldf
from polars.io._utils import get_sources
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)
from polars.io.scan_options._options import ScanOptions

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import PyLazyFrame

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import StorageOptionsDict
    from polars.dataframe.frame import DataFrame
    from polars.io.cloud import CredentialProviderFunction
    from polars.lazyframe.frame import LazyFrame


@unstable()
def read_lines(
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
    name: str = "lines",
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    glob: bool = True,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    include_file_paths: str | None = None,
) -> DataFrame:
    r"""
    Read lines into a string column from a file.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    source
        Path(s) to a file or directory
        When needing to authenticate for scanning cloud locations, see the
        `storage_options` parameter.
    name
        Name to use for the output column.
    n_rows
        Stop reading from parquet file after reading `n_rows`.
    row_index_name
        If not None, this will insert a row index column with the given name into the
        DataFrame
    row_index_offset
        Offset to start the row index column (only used if the name is set)
    glob
        Expand path given via globbing rules.
    storage_options
        Options that indicate how to connect to a cloud provider.

        The cloud providers currently supported are AWS, GCP, and Azure.
        See supported keys here:

        * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
        * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
        * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_
        * Hugging Face (`hf://`): Accepts an API key under the `token` parameter: \
          `{'token': '...'}`, or by setting the `HF_TOKEN` environment variable.

        If `storage_options` is not provided, Polars will try to infer the information
        from environment variables.
    credential_provider
        Provide a function that can be called to provide cloud storage
        credentials. The function is expected to return a dictionary of
        credential keys along with an optional credential expiry time.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    include_file_paths
        Include the path of the source file(s) as a column with this name.

    See Also
    --------
    scan_lines

    Examples
    --------
    >>> pl.read_lines(b"Hello\nworld")
    shape: (2, 1)
    ┌───────┐
    │ lines │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ Hello │
    │ world │
    └───────┘
    """
    return scan_lines(
        source,
        name=name,
        n_rows=n_rows,
        row_index_name=row_index_name,
        row_index_offset=row_index_offset,
        glob=glob,
        storage_options=storage_options,
        credential_provider=credential_provider,
        include_file_paths=include_file_paths,
    ).collect()


@unstable()
def scan_lines(
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
    name: str = "lines",
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    glob: bool = True,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    include_file_paths: str | None = None,
) -> LazyFrame:
    r"""
    Construct a LazyFrame which scans lines into a string column from a file.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    source
        Path(s) to a file or directory
        When needing to authenticate for scanning cloud locations, see the
        `storage_options` parameter.
    name
        Name to use for the output column.
    n_rows
        Stop reading from parquet file after reading `n_rows`.
    row_index_name
        If not None, this will insert a row index column with the given name into the
        DataFrame
    row_index_offset
        Offset to start the row index column (only used if the name is set)
    glob
        Expand path given via globbing rules.
    storage_options
        Options that indicate how to connect to a cloud provider.

        The cloud providers currently supported are AWS, GCP, and Azure.
        See supported keys here:

        * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
        * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
        * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_
        * Hugging Face (`hf://`): Accepts an API key under the `token` parameter: \
          `{'token': '...'}`, or by setting the `HF_TOKEN` environment variable.

        If `storage_options` is not provided, Polars will try to infer the information
        from environment variables.
    credential_provider
        Provide a function that can be called to provide cloud storage
        credentials. The function is expected to return a dictionary of
        credential keys along with an optional credential expiry time.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    include_file_paths
        Include the path of the source file(s) as a column with this name.

    See Also
    --------
    read_lines

    Examples
    --------
    >>> pl.scan_lines(b"Hello\nworld").collect()
    shape: (2, 1)
    ┌───────┐
    │ lines │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ Hello │
    │ world │
    └───────┘
    """
    sources = get_sources(source)

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, sources, storage_options, "scan_lines"
    )
    del credential_provider

    pylf = PyLazyFrame.new_from_scan_lines(
        sources=sources,
        scan_options=ScanOptions(
            row_index=(
                (row_index_name, row_index_offset)
                if row_index_name is not None
                else None
            ),
            pre_slice=(0, n_rows) if n_rows is not None else None,
            include_file_paths=include_file_paths,
            glob=glob,
            storage_options=storage_options,
            credential_provider=credential_provider_builder,
        ),
        name=name,
    )

    return wrap_ldf(pylf)
