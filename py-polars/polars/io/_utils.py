from __future__ import annotations

import glob
import re
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import IO, Any, ContextManager, Iterator, Sequence, overload

from polars._utils.various import is_int_sequence, is_str_sequence, normalize_filepath
from polars.dependencies import _FSSPEC_AVAILABLE, fsspec
from polars.exceptions import NoDataError


def parse_columns_arg(
    columns: Sequence[str] | Sequence[int] | str | int | None,
) -> tuple[Sequence[int] | None, Sequence[str] | None]:
    """
    Parse the `columns` argument of an I/O function.

    Disambiguates between column names and column indices input.

    Returns
    -------
    tuple
        A tuple containing the columns as a projection and a list of column names.
        Only one will be specified, the other will be `None`.
    """
    if columns is None:
        return None, None

    projection: Sequence[int] | None = None
    column_names: Sequence[str] | None = None

    if isinstance(columns, str):
        column_names = [columns]
    elif isinstance(columns, int):
        projection = [columns]
    elif is_str_sequence(columns):
        _ensure_columns_are_unique(columns)
        column_names = columns
    elif is_int_sequence(columns):
        _ensure_columns_are_unique(columns)
        projection = columns
    else:
        msg = "the `columns` argument should contain a list of all integers or all string values"
        raise TypeError(msg)

    return projection, column_names


def _ensure_columns_are_unique(columns: Sequence[str] | Sequence[int]) -> None:
    if len(columns) != len(set(columns)):
        msg = f"`columns` arg should only have unique values, got {columns!r}"
        raise ValueError(msg)


def parse_row_index_args(
    row_index_name: str | None = None,
    row_index_offset: int = 0,
) -> tuple[str, int] | None:
    """
    Parse the `row_index_name` and `row_index_offset` arguments of an I/O function.

    The Rust functions take a single tuple rather than two separate arguments.
    """
    if row_index_name is None:
        return None
    else:
        return (row_index_name, row_index_offset)


def _replace(file: str, replace_chars_map: list[tuple[str, str]]) -> str:
    for old, new in replace_chars_map:
        file = file.replace(old, new)
    return file


@overload
def prepare_file_arg(
    file: str | Path | list[str] | IO[bytes] | bytes,
    encoding: str | None = ...,
    *,
    use_pyarrow: bool = ...,
    raise_if_empty: bool = ...,
    storage_options: dict[str, Any] | None = ...,
    replace_chars_map: list[tuple[str, str]] | None = None,
) -> ContextManager[str | BytesIO]: ...


@overload
def prepare_file_arg(
    file: str | Path | IO[str] | IO[bytes] | bytes,
    encoding: str | None = ...,
    *,
    use_pyarrow: bool = ...,
    raise_if_empty: bool = ...,
    storage_options: dict[str, Any] | None = ...,
    replace_chars_map: list[tuple[str, str]] | None = None,
) -> ContextManager[str | BytesIO]: ...


@overload
def prepare_file_arg(
    file: str | Path | list[str] | IO[str] | IO[bytes] | bytes,
    encoding: str | None = ...,
    *,
    use_pyarrow: bool = ...,
    raise_if_empty: bool = ...,
    storage_options: dict[str, Any] | None = ...,
    replace_chars_map: list[tuple[str, str]] | None = None,
) -> ContextManager[str | list[str] | BytesIO | list[BytesIO]]: ...


def prepare_file_arg(
    file: str | Path | list[str] | IO[str] | IO[bytes] | bytes,
    encoding: str | None = None,
    *,
    use_pyarrow: bool = False,
    raise_if_empty: bool = True,
    storage_options: dict[str, Any] | None = None,
    replace_chars_map: list[tuple[str, str]] | None = None,
) -> ContextManager[str | list[str] | BytesIO | list[BytesIO]]:
    """
    Prepare file argument.

    Utility for read_[csv, parquet]. (not to be used by scan_[csv, parquet]).
    Returned value is always usable as a context.

    A `StringIO`, `BytesIO` file is returned as a `BytesIO`.
    A local path is returned as a string.
    An http URL is read into a buffer and returned as a `BytesIO`.

    When `encoding` is not `utf8` or `utf8-lossy`, the whole file is
    first read in Python and decoded using the specified encoding and
    returned as a `BytesIO` (for usage with `read_csv`).

    A `bytes` file is returned as a `BytesIO` if `use_pyarrow=True`.

    When fsspec is installed, remote file(s) is (are) opened with
    `fsspec.open(file, **kwargs)` or `fsspec.open_files(file, **kwargs)`.
    If encoding is not `utf8` or `utf8-lossy`, decoding is handled by
    fsspec too.
    """
    if replace_chars_map is None:
        replace_chars_map = []
    storage_options = storage_options.copy() if storage_options else {}
    if storage_options and not _FSSPEC_AVAILABLE:
        msg = "`fsspec` is required for `storage_options` argument"
        raise ImportError(msg)

    # Small helper to use a variable as context
    @contextmanager
    def managed_file(file: Any) -> Iterator[Any]:
        try:
            yield file
        finally:
            pass

    has_utf8_utf8_lossy_encoding = (
        encoding in {"utf8", "utf8-lossy"} if encoding else True
    )
    encoding_str = encoding if encoding else "utf8"

    # PyArrow allows directories, so we only check that something is not
    # a dir if we are not using PyArrow
    check_not_dir = not use_pyarrow

    if isinstance(file, bytes):
        if not has_utf8_utf8_lossy_encoding or len(replace_chars_map) > 0:
            file = _replace(file.decode(encoding_str), replace_chars_map).encode("utf8")
        return _check_empty(
            BytesIO(file), context="bytes", raise_if_empty=raise_if_empty
        )

    if isinstance(file, StringIO):
        return _check_empty(
            BytesIO(_replace(file.read(), replace_chars_map).encode("utf8")),
            context="StringIO",
            read_position=file.tell(),
            raise_if_empty=raise_if_empty,
        )

    if isinstance(file, BytesIO):
        if not has_utf8_utf8_lossy_encoding or len(replace_chars_map) > 0:
            return _check_empty(
                BytesIO(
                    _replace(
                        file.read().decode(encoding_str), replace_chars_map
                    ).encode("utf8")
                ),
                context="BytesIO",
                read_position=file.tell(),
                raise_if_empty=raise_if_empty,
            )
        return managed_file(
            _check_empty(
                b=file,
                context="BytesIO",
                read_position=file.tell(),
                raise_if_empty=raise_if_empty,
            )
        )

    if isinstance(file, Path):
        if not has_utf8_utf8_lossy_encoding or len(replace_chars_map) > 0:
            return _check_empty(
                BytesIO(
                    _replace(
                        file.read_bytes().decode(encoding_str), replace_chars_map
                    ).encode("utf8")
                ),
                context=f"Path ({file!r})",
                raise_if_empty=raise_if_empty,
            )
        return managed_file(normalize_filepath(file, check_not_directory=check_not_dir))

    if isinstance(file, str):
        # make sure that this is before fsspec
        # as fsspec needs requests to be installed
        # to read from http
        if looks_like_url(file):
            return process_file_url(file, encoding_str, replace_chars_map)
        if _FSSPEC_AVAILABLE:
            from fsspec.utils import infer_storage_options

            # check if it is a local file
            if infer_storage_options(file)["protocol"] == "file":
                # (lossy) utf8
                if has_utf8_utf8_lossy_encoding and len(replace_chars_map) == 0:
                    return managed_file(
                        normalize_filepath(file, check_not_directory=check_not_dir)
                    )
                # decode first
                with Path(file).open(encoding=encoding_str) as f:
                    return _check_empty(
                        BytesIO(_replace(f.read(), replace_chars_map).encode("utf8")),
                        context=f"{file!r}",
                        raise_if_empty=raise_if_empty,
                    )

            storage_options["encoding"] = encoding
            if len(replace_chars_map) == 0:
                return fsspec.open(file, **storage_options)
            else:
                with fsspec.open(file, **storage_options, mode="rb") as f:
                    fread: bytes = f.read()  # type: ignore[assignment]
                    encoding = encoding or "utf8"
                    return _check_empty(
                        BytesIO(
                            _replace(fread.decode(encoding), replace_chars_map).encode(
                                "utf8"
                            )
                        ),
                        context=f"{file!r}",
                        raise_if_empty=raise_if_empty,
                    )

    if isinstance(file, list) and bool(file) and all(isinstance(f, str) for f in file):
        if _FSSPEC_AVAILABLE:
            from fsspec.utils import infer_storage_options

            if has_utf8_utf8_lossy_encoding:
                if all(infer_storage_options(f)["protocol"] == "file" for f in file):
                    return managed_file(
                        [
                            normalize_filepath(f, check_not_directory=check_not_dir)
                            for f in file
                        ]
                    )
            storage_options["encoding"] = encoding
            return fsspec.open_files(file, **storage_options)

    if isinstance(file, str):
        file = normalize_filepath(file, check_not_directory=check_not_dir)
        if not has_utf8_utf8_lossy_encoding or len(replace_chars_map) > 0:
            with Path(file).open(encoding=encoding_str) as f:
                return _check_empty(
                    BytesIO(_replace(f.read(), replace_chars_map).encode("utf8")),
                    context=f"{file!r}",
                    raise_if_empty=raise_if_empty,
                )

    return managed_file(file)


def _check_empty(
    b: BytesIO, *, context: str, raise_if_empty: bool, read_position: int | None = None
) -> BytesIO:
    if raise_if_empty and b.getbuffer().nbytes == 0:
        hint = (
            f" (buffer position = {read_position}; try seek(0) before reading?)"
            if context in ("StringIO", "BytesIO") and read_position
            else ""
        )
        msg = f"empty CSV data from {context}{hint}"
        raise NoDataError(msg)
    return b


def looks_like_url(path: str) -> bool:
    return re.match("^(ht|f)tps?://", path, re.IGNORECASE) is not None


def process_file_url(
    path: str,
    encoding: str | None = None,
    replace_chars_map: list[tuple[str, str]] | None = None,
) -> BytesIO:
    if replace_chars_map is None:
        replace_chars_map = []
    from urllib.request import urlopen

    with urlopen(path) as f:
        if (
            not encoding
            or encoding in {"utf8", "utf8-lossy"}
            and len(replace_chars_map) == 0
        ):
            return BytesIO(f.read())
        else:
            encoding = encoding = "utf8"
            return BytesIO(
                _replace(f.read().decode(encoding), replace_chars_map).encode("utf8")
            )


def is_glob_pattern(file: str) -> bool:
    return any(char in file for char in ["*", "?", "["])


def is_local_file(file: str) -> bool:
    try:
        next(glob.iglob(file, recursive=True))  # noqa: PTH207
    except StopIteration:
        return False
    else:
        return True
