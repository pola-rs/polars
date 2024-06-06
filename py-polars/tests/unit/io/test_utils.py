from __future__ import annotations

from typing import Sequence

import pytest

from polars.io._utils import looks_like_url, parse_columns_arg, parse_row_index_args


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
