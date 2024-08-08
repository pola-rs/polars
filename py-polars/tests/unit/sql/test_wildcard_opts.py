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
            "ID": [333, 666, 999],
            "FirstName": ["Bruce", "Diana", "Clark"],
            "LastName": ["Wayne", "Prince", "Kent"],
            "Address": ["Batcave", "Paradise Island", "Fortress of Solitude"],
            "City": ["Gotham", "Themyscira", "Metropolis"],
        }
    )


@pytest.mark.parametrize(
    ("excluded", "order_by", "expected"),
    [
        ("ID", "ORDER BY 2, 1", ["FirstName", "LastName", "Address", "City"]),
        ("(ID)", "ORDER BY City", ["FirstName", "LastName", "Address", "City"]),
        ("(Address, LastName, FirstName)", "", ["ID", "City"]),
        ('("ID", "FirstName", "LastName", "Address", "City")', "", []),
    ],
)
def test_select_exclude(
    excluded: str,
    order_by: str,
    expected: list[str],
    df: pl.DataFrame,
) -> None:
    for exclude_keyword in ("EXCLUDE", "EXCEPT"):
        assert (
            df.sql(f"SELECT * {exclude_keyword} {excluded} FROM self").columns
            == expected
        )


def test_select_exclude_order_by(
    df: pl.DataFrame,
) -> None:
    expected = pl.DataFrame(
        {
            "FirstName": ["Diana", "Clark", "Bruce"],
            "Address": ["Paradise Island", "Fortress of Solitude", "Batcave"],
        }
    )
    for order_by in ("", "ORDER BY 1 DESC", "ORDER BY 2 DESC", "ORDER BY Address DESC"):
        actual = df.sql(f"SELECT * EXCLUDE (ID,LastName,City) FROM self {order_by}")
        if not order_by:
            actual = actual.sort("FirstName", descending=True)
        assert_frame_equal(actual, expected)


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


@pytest.mark.parametrize("order_by", ["1 DESC", "Name DESC", "FirstName DESC"])
def test_select_rename_exclude_sort(order_by: str, df: pl.DataFrame) -> None:
    actual = df.sql(
        f"""
        SELECT * EXCLUDE (ID, City, LastName) RENAME FirstName AS Name
        FROM self
        ORDER BY {order_by}
        """
    )
    expected = pl.DataFrame(
        {
            "Name": ["Diana", "Clark", "Bruce"],
            "Address": ["Paradise Island", "Fortress of Solitude", "Batcave"],
        }
    )
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("replacements", "check_cols", "expected"),
    [
        (
            "(ID // 3 AS ID)",
            ["ID"],
            [(333,), (222,), (111,)],
        ),
        (
            "(ID // 3 AS ID) RENAME (ID AS Identifier)",
            ["Identifier"],
            [(333,), (222,), (111,)],
        ),
        (
            "((City || ':' || City) AS City, ID // -3 AS ID)",
            ["City", "ID"],
            [
                ("Gotham:Gotham", -111),
                ("Themyscira:Themyscira", -222),
                ("Metropolis:Metropolis", -333),
            ],
        ),
    ],
)
def test_select_replace(
    replacements: str,
    check_cols: list[str],
    expected: list[tuple[Any]],
    df: pl.DataFrame,
) -> None:
    for order_by in ("", "ORDER BY ID DESC", "ORDER BY -ID ASC"):
        res = df.sql(f"SELECT * REPLACE {replacements} FROM self {order_by}")
        if not order_by:
            res = res.sort(check_cols[-1], descending=True)

        assert res.select(check_cols).rows() == expected
        expected_columns = (
            check_cols + df.columns[1:] if check_cols == ["Identifier"] else df.columns
        )
        assert res.columns == expected_columns


def test_select_wildcard_errors(df: pl.DataFrame) -> None:
    # EXCLUDE and ILIKE are not allowed together
    with pytest.raises(SQLInterfaceError, match="ILIKE"):
        assert df.sql("SELECT * EXCLUDE Address ILIKE '%o%' FROM self")

    # these two options are aliases, with EXCLUDE being preferred
    with pytest.raises(
        SQLInterfaceError,
        match="EXCLUDE and EXCEPT wildcard options cannot be used together",
    ):
        assert df.sql("SELECT * EXCLUDE Address EXCEPT City FROM self")

    # note: missing "()" around the exclude option results in dupe col
    with pytest.raises(
        DuplicateError,
        match="the name 'City' is duplicate",
    ):
        assert df.sql("SELECT * EXCLUDE Address, City FROM self")
