from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

from polars._utils.various import is_path_or_str_sequence, normalize_filepath
from polars._utils.wrap import wrap_df, wrap_ldf
from polars.io._utils import parse_columns_arg, parse_row_index_args
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyDataFrame, PyLazyFrame

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame
    from polars._typing import FileSource
    from polars.io.cloud import CredentialProviderFunction


def read_avro(
    source: str | Path | IO[bytes] | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
) -> DataFrame:
    """
    Read into a DataFrame from Apache Avro format.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance). For file-like objects, the stream position
        may not be updated accordingly after reading.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from Apache Avro file after reading `n_rows`.

    Returns
    -------
    DataFrame
    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
    projection, column_names = parse_columns_arg(columns)

    pydf = PyDataFrame.read_avro(source, column_names, projection, n_rows)
    return wrap_df(pydf)


def scan_avro(
    source: FileSource,
    *,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = False,
    storage_options: dict[str, Any] | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    retries: int = 2,
    include_file_paths: str | None = None,
    file_cache_ttl: int | None = None,
) -> LazyFrame:
    """Lazily read from a local or cloud-hosted avro file (or files).

    This function allows the query optimizer to push down predicates and projections to
    the scan level, typically increasing performance and reducing memory overhead.

    Parameters
    ----------
    source
        Path(s) to a file or directory
        When needing to authenticate for scanning cloud locations, see the
        `storage_options` parameter.
    n_rows
        Stop reading from avro file after reading `n_rows`.
    row_index_name
        If not None, this will insert a row index column with the given name into the
        DataFrame
    row_index_offset
        Offset to start the row index column (only used if the name is set)
    rechunk
        In case of reading multiple files via a glob pattern rechunk the final DataFrame
        into contiguous memory chunks.
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
    retries
        Number of retries if accessing a cloud instance fails.
    include_file_paths
        Include the path of the source file(s) as a column with this name.
    file_cache_ttl
        Amount of time to keep downloaded cloud files since their last access time,
        in seconds. Uses the `POLARS_FILE_CACHE_TTL` environment variable
        (which defaults to 1 hour) if not given.

    See Also
    --------
    read_avro

    Examples
    --------
    Scan a local avro file.

    >>> pl.scan_avro("path/to/file.avro")  # doctest: +SKIP

    Scan a file on AWS S3.

    >>> source = "s3://bucket/*.avro"
    >>> pl.scan_avro(source)  # doctest: +SKIP
    >>> storage_options = {
    ...     "aws_access_key_id": "<secret>",
    ...     "aws_secret_access_key": "<secret>",
    ...     "aws_region": "us-east-1",
    ... }
    >>> pl.scan_avro(source, storage_options=storage_options)  # doctest: +SKIP
    """
    single_source: IO[bytes] | bytes | str | None
    sources: list[str] | list[Path] | list[IO[bytes]] | list[bytes]
    if isinstance(source, (str, Path)):
        single_source = normalize_filepath(source, check_not_directory=False)
        sources = []
    elif is_path_or_str_sequence(source):
        sources = [
            normalize_filepath(source, check_not_directory=False) for source in source
        ]
        single_source = None
    elif isinstance(source, list):
        single_source = None
        sources = source
    else:
        single_source = source
        sources = []

    credential_provider_builder = _init_credential_provider_builder(
        credential_provider, source, storage_options, "scan_avro"
    )

    if storage_options:
        storage_options = list(storage_options.items())  # type: ignore[assignment]
    else:
        # Handle empty dict input
        storage_options = None

    pylf = PyLazyFrame.new_from_avro(
        single_source,
        sources,
        n_rows,
        rechunk,
        parse_row_index_args(row_index_name, row_index_offset),
        cloud_options=storage_options,
        credential_provider=credential_provider_builder,
        retries=retries,
        file_cache_ttl=file_cache_ttl,
        include_file_paths=include_file_paths,
    )
    return wrap_ldf(pylf)
