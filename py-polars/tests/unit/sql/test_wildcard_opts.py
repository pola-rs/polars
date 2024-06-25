from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import DuplicateError


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame({"num": [999, 666], "str": ["b", "a"], "val": [2.0, 0.5]})


@pytest.mark.parametrize(
    ("excluded", "expected"),
    [
        ("num", ["str", "val"]),
        ("(val, num)", ["str"]),
        ("(str, num)", ["val"]),
        ("(str, val, num)", []),
    ],
)
def test_select_exclude(
    excluded: str,
    expected: list[str],
    df: pl.DataFrame,
) -> None:
    assert df.sql(f"SELECT * EXCLUDE {excluded} FROM self").columns == expected


def test_select_exclude_error(df: pl.DataFrame) -> None:
    with pytest.raises(DuplicateError, match="the name 'num' is duplicate"):
        # note: missing "()" around the exclude option results in dupe col
        assert df.sql("SELECT * EXCLUDE val, num FROM self")


@pytest.mark.parametrize(
    ("renames", "expected"),
    [
        ("val AS value", ["num", "str", "value"]),
        ("(num AS flt)", ["flt", "str", "val"]),
        ("(val AS value, num AS flt)", ["flt", "str", "value"]),
    ],
)
def test_select_rename(
    renames: str,
    expected: list[str],
    df: pl.DataFrame,
) -> None:
    assert df.sql(f"SELECT * RENAME {renames} FROM self").columns == expected


@pytest.mark.parametrize(
    ("replacements", "check_cols", "expected"),
    [
        (
            "(num // 3 AS num)",
            ["num"],
            [(333,), (222,)],
        ),
        (
            "((str || str) AS str, num / 3 AS num)",
            ["num", "str"],
            [(333, "bb"), (222, "aa")],
        ),
    ],
)
def test_select_replace(
    replacements: str,
    check_cols: list[str],
    expected: list[tuple[Any]],
    df: pl.DataFrame,
) -> None:
    res = df.sql(f"SELECT * REPLACE {replacements} FROM self")

    assert res.select(check_cols).rows() == expected
    assert res.columns == df.columns
