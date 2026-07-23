from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars._utils.various import is_path_or_str_sequence, normalize_filepath
from polars.io._utils import (
    get_sources,
    looks_like_url,
    parse_columns_arg,
    parse_row_index_args,
)
from polars.io.cloud._utils import _first_scan_path, _get_path_scheme
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)
from tests.unit.utils.pathlike import HostilePathLike

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["a", "b"], (None, ["a", "b"])),
        ((1, 2), ((1, 2), None)),
        ("foo", (None, ["foo"])),
        (3, ([3], None)),
        (None, (None, None)),
    ],
)
def test_parse_columns_arg(
    columns: Sequence[str] | Sequence[int] | str | int | None,
    expected: tuple[Sequence[int] | None, Sequence[str] | None],
) -> None:
    assert parse_columns_arg(columns) == expected


def test_parse_columns_arg_mixed_types() -> None:
    with pytest.raises(TypeError):
        parse_columns_arg(["a", 1])  # type: ignore[arg-type]


@pytest.mark.parametrize("columns", [["a", "a"], [1, 1, 2]])
def test_parse_columns_arg_duplicates(columns: Sequence[str] | Sequence[int]) -> None:
    with pytest.raises(ValueError):
        parse_columns_arg(columns)


def test_parse_row_index_args() -> None:
    assert parse_row_index_args("idx", 5) == ("idx", 5)
    assert parse_row_index_args(None, 5) is None


@pytest.mark.parametrize(
    ("url", "result"),
    [
        ("HTTPS://pola.rs/data.csv", True),
        ("http://pola.rs/data.csv", True),
        ("ftps://pola.rs/data.csv", True),
        ("FTP://pola.rs/data.csv", True),
        ("htp://pola.rs/data.csv", False),
        ("fttp://pola.rs/data.csv", False),
        ("http_not_a_url", False),
        ("ftp_not_a_url", False),
        ("/mnt/data.csv", False),
        ("file://mnt/data.csv", False),
    ],
)
def test_looks_like_url(url: str, result: bool) -> None:
    assert looks_like_url(url) == result


@pytest.mark.parametrize(
    "scan", [pl.scan_csv, pl.scan_parquet, pl.scan_ndjson, pl.scan_ipc]
)
def test_filename_in_err(scan: Any) -> None:
    with pytest.raises(FileNotFoundError, match=r".*does not exist"):
        scan("does not exist").collect()


def test_get_path_scheme() -> None:
    assert _get_path_scheme("") is None
    assert _get_path_scheme("A") is None
    assert _get_path_scheme("scheme://") == "scheme"
    assert _get_path_scheme("://") == ""
    assert _get_path_scheme("://...") == ""


def test_get_path_scheme_os_pathlike_17828() -> None:
    # Must use `os.fspath`, not `str`, on the path-like.
    assert _get_path_scheme(HostilePathLike("scheme://bucket/key")) == "scheme"
    assert _get_path_scheme(HostilePathLike("/local/path")) is None


def test_normalize_filepath_os_pathlike_17828() -> None:
    # The path-like is resolved via `os.fspath`, yielding a plain string.
    result = normalize_filepath(HostilePathLike("/some/path"))
    assert result == "/some/path"
    assert isinstance(result, str)


def test_is_path_or_str_sequence_os_pathlike_17828() -> None:
    assert is_path_or_str_sequence([HostilePathLike("a"), Path("b"), "c"])
    assert is_path_or_str_sequence([HostilePathLike("a")])
    # A single path-like is not a sequence.
    assert not is_path_or_str_sequence(HostilePathLike("a"))
    # A non-path element disqualifies the sequence.
    assert not is_path_or_str_sequence([HostilePathLike("a"), 123])


def test_get_sources_os_pathlike_17828() -> None:
    # A single path-like is normalized to a one-element list of strings.
    assert get_sources(HostilePathLike("/a/b")) == ["/a/b"]
    # A sequence of path-likes is normalized element-wise.
    sources: list[os.PathLike[str]] = [HostilePathLike("/a/b"), Path("/c/d")]
    assert get_sources(sources) == ["/a/b", "/c/d"]


def test_first_scan_path_os_pathlike_17828() -> None:
    single = HostilePathLike("/a/b")
    assert _first_scan_path(single) is single

    seq = [HostilePathLike("/a/b"), HostilePathLike("/c/d")]
    assert _first_scan_path(seq) is seq[0]


@pytest.mark.parametrize(
    "uri", ["s3://bucket/data.parquet", "az://container/data.parquet"]
)
def test_init_credential_provider_builder_os_pathlike_17828(uri: str) -> None:
    # The AWS/Azure branches resolve the path with `os.fspath`, not `str`.
    builder = _init_credential_provider_builder(
        "auto", HostilePathLike(uri), None, "scan_parquet"
    )
    assert builder is not None
