from __future__ import annotations

import glob
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, BinaryIO, ContextManager, Iterator, TextIO, overload
from urllib.request import urlopen

from polars.datatypes import DataType
from polars.utils import format_path

try:
    import fsspec
    from fsspec.utils import infer_storage_options

    _WITH_FSSPEC = True
except ImportError:
    _WITH_FSSPEC = False

try:
    from polars.polars import ipc_schema as _ipc_schema
    from polars.polars import parquet_schema as _parquet_schema
except ImportError:
    pass


def _process_http_file(path: str) -> BytesIO:
    with urlopen(path) as f:
        return BytesIO(f.read())


@overload
def _prepare_file_arg(
    file: str | list[str] | Path | BinaryIO | bytes, **kwargs: Any
) -> ContextManager[str | BinaryIO]:
    ...


@overload
def _prepare_file_arg(
    file: str | TextIO | Path | BinaryIO | bytes, **kwargs: Any
) -> ContextManager[str | BinaryIO]:
    ...


@overload
def _prepare_file_arg(
    file: str | list[str] | TextIO | Path | BinaryIO | bytes, **kwargs: Any
) -> ContextManager[str | list[str] | BinaryIO | list[BinaryIO]]:
    ...


def _prepare_file_arg(
    file: str | list[str] | TextIO | Path | BinaryIO | bytes, **kwargs: Any
) -> ContextManager[str | BinaryIO | list[str] | list[BinaryIO]]:
    """
    Prepare file argument.

    Utility for read_[csv, parquet]. (not to be used by scan_[csv, parquet]).
    Returned value is always usable as a context.

    A `StringIO`, `BytesIO` file is returned as a `BytesIO`.
    A local path is returned as a string.
    An http URL is read into a buffer and returned as a `BytesIO`.

    When fsspec is installed, remote file(s) is (are) opened with
    `fsspec.open(file, **kwargs)` or `fsspec.open_files(file, **kwargs)`.

    """
    # Small helper to use a variable as context
    @contextmanager
    def managed_file(file: Any) -> Iterator[Any]:
        try:
            yield file
        finally:
            pass

    if isinstance(file, StringIO):
        return BytesIO(file.read().encode("utf8"))
    if isinstance(file, BytesIO):
        return managed_file(file)
    if isinstance(file, Path):
        return managed_file(format_path(file))
    if isinstance(file, str):
        # make sure that this is before fsspec
        # as fsspec needs requests to be installed
        # to read from http
        if file.startswith("http"):
            return _process_http_file(file)
        if _WITH_FSSPEC:
            if infer_storage_options(file)["protocol"] == "file":
                return managed_file(format_path(file))
            return fsspec.open(file, **kwargs)
    if isinstance(file, list) and bool(file) and all(isinstance(f, str) for f in file):
        if _WITH_FSSPEC:
            if all(infer_storage_options(f)["protocol"] == "file" for f in file):
                return managed_file([format_path(f) for f in file])
            return fsspec.open_files(file, **kwargs)
    if isinstance(file, str):
        file = format_path(file)
    return managed_file(file)


def read_ipc_schema(file: str | BinaryIO | Path | bytes) -> dict[str, type[DataType]]:
    """
    Get a schema of the IPC file without reading data.

    Parameters
    ----------
    file
        Path to a file or a file-like object.

    Returns
    -------
    Dictionary mapping column names to datatypes

    """
    if isinstance(file, (str, Path)):
        file = format_path(file)

    return _ipc_schema(file)


def read_parquet_schema(
    file: str | BinaryIO | Path | bytes,
) -> dict[str, type[DataType]]:
    """
    Get a schema of the Parquet file without reading data.

    Parameters
    ----------
    file
        Path to a file or a file-like object.

    Returns
    -------
    Dictionary mapping column names to datatypes

    """
    if isinstance(file, (str, Path)):
        file = format_path(file)

    return _parquet_schema(file)


def _is_local_file(file: str) -> bool:
    try:
        next(glob.iglob(file, recursive=True))
        return True
    except StopIteration:
        return False
