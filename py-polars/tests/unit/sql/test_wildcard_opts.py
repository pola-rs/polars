from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import DuplicateError, SQLInterfaceError
from polars.testing import assert_frame_equal


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ID": [333, 999],
            "FirstName": ["Bruce", "Clark"],
            "LastName": ["Wayne", "Kent"],
            "Address": ["The Batcave", "Fortress of Solitude"],
            "City": ["Gotham", "Metropolis"],
        }
    )


@pytest.mark.parametrize(
    ("excluded", "expected"),
    [
        ("ID", ["FirstName", "LastName", "Address", "City"]),
        ("(ID)", ["FirstName", "LastName", "Address", "City"]),
        ("(Address, LastName, FirstName)", ["ID", "City"]),
        ('("ID", "FirstName", "LastName", "Address", "City")', []),
    ],
)
def test_select_exclude(
    excluded: str,
    expected: list[str],
    df: pl.DataFrame,
) -> None:
    assert df.sql(f"SELECT * EXCLUDE {excluded} FROM self").columns == expected


def test_select_exclude_order_by(
    df: pl.DataFrame,
) -> None:
    expected = pl.DataFrame(
        {
            "FirstName": ["Clark", "Bruce"],
            "Address": ["Fortress of Solitude", "The Batcave"],
        }
    )
    for order_by in ("ORDER BY 2", "ORDER BY 1 DESC"):
        actual = df.sql(f"SELECT * EXCLUDE (ID,LastName,City) FROM self {order_by}")
        assert_frame_equal(actual, expected)


def test_select_exclude_error(df: pl.DataFrame) -> None:
    # EXCLUDE and ILIKE are not allowed together
    with pytest.raises(SQLInterfaceError, match="ILIKE"):
        assert df.sql("SELECT * EXCLUDE Address ILIKE '%o%' FROM self")

    # note: missing "()" around the exclude option results in dupe col
    with pytest.raises(DuplicateError, match="the name 'City' is duplicate"):
        assert df.sql("SELECT * EXCLUDE Address, City FROM self")


def test_ilike(df: pl.DataFrame) -> None:
    assert df.sql("SELECT * ILIKE 'a%e' FROM self").columns == []
    assert df.sql("SELECT * ILIKE '%nam_' FROM self").columns == [
        "FirstName",
        "LastName",
    ]
    assert df.sql("SELECT * ILIKE '%a%e%' FROM self").columns == [
        "FirstName",
        "LastName",
        "Address",
    ]
    assert df.sql(
        """SELECT * ILIKE '%I%' RENAME (FirstName AS Name) FROM self"""
    ).columns == [
        "ID",
        "Name",
        "City",
    ]


@pytest.mark.parametrize(
    ("renames", "expected"),
    [
        (
            "Address AS Location",
            ["ID", "FirstName", "LastName", "Location", "City"],
        ),
        (
            '(Address AS "Location")',
            ["ID", "FirstName", "LastName", "Location", "City"],
        ),
        (
            '("Address" AS Location, "ID" AS PersonID)',
            ["PersonID", "FirstName", "LastName", "Location", "City"],
        ),
    ],
)
def test_select_rename(
    renames: str,
    expected: list[str],
    df: pl.DataFrame,
) -> None:
    assert df.sql(f"SELECT * RENAME {renames} FROM self").columns == expected


@pytest.mark.parametrize(
    ("replacements", "order_by", "check_cols", "expected"),
    [
        (
            "(ID // 3 AS ID)",
            "",
            ["ID"],
            [(111,), (333,)],
        ),
        (
            "((City || ':' || City) AS City, ID // 3 AS ID)",
            "ORDER BY ID DESC",
            ["City", "ID"],
            [("Metropolis:Metropolis", 333), ("Gotham:Gotham", 111)],
        ),
    ],
)
def test_select_replace(
    replacements: str,
    order_by: str,
    check_cols: list[str],
    expected: list[tuple[Any]],
    df: pl.DataFrame,
) -> None:
    res = df.sql(f"SELECT * REPLACE {replacements} FROM self {order_by}")

    assert res.select(check_cols).rows() == expected
    assert res.columns == df.columns
