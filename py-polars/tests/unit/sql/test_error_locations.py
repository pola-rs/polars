from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, SQLInterfaceError


@pytest.mark.parametrize(
    ("query", "error", "location"),
    [
        (
            "SELECT a,\n  LEN(s)\nFROM df",
            SQLInterfaceError,
            "SQL at line 2, column 3:\nLINE 2:   LEN(s)\n          ^",
        ),
        (
            "SELECT * FROM nope",
            SQLInterfaceError,
            "SQL at line 1, column 15:\nLINE 1: SELECT * FROM nope\n                      ^",
        ),
        (
            "SELECT * FROM df NATURAL JOIN other",
            SQLInterfaceError,
            "In the JOIN clause at line 1, column 31.",
        ),
        (
            "-- the order\nSELECT a\nFROM df ORDER BY a + s",
            InvalidOperationError,
            "In the SQL statement: SELECT a FROM df ORDER BY a + s",
        ),
    ],
)
def test_error_location(query: str, error: type[Exception], location: str) -> None:
    df = pl.DataFrame({"a": [1, 2], "s": ["x", "y"]})
    other = pl.DataFrame({"b": [1]})
    with pytest.raises(error) as exc:
        pl.sql(query, eager=True)
    message = str(exc.value)
    assert f"\n\n{location}" in message
    plan = message.find("Resolved plan")
    assert plan == -1 or message.index(location) < plan


def test_error_location_on_a_long_line() -> None:
    df = pl.DataFrame({"a": [1], "s": ["x"]})
    columns = ", ".join(f"a AS c{i}" for i in range(40))
    with pytest.raises(SQLInterfaceError) as exc:
        pl.sql(f"SELECT {columns}, LEN(s), {columns} FROM df", eager=True)
    line, caret = str(exc.value).split("\n\n")[1].split("\n")[1:3]
    assert line.startswith("LINE 1: ...")
    assert line.endswith("...")
    assert len(line) <= len("LINE 1: ") + 106
    assert line[caret.index("^") :].startswith("LEN(s)")


def test_error_hint() -> None:
    with pytest.raises(SQLInterfaceError) as exc:
        pl.sql("SELECT LEN('x')", eager=True)
    message = str(exc.value)
    # The location, then the hint, then the plan.
    location = message.index("\n\nSQL at line 1, column 8:")
    hint = message.index("\n\nHint: use LENGTH instead")
    plan = message.find("Resolved plan")
    assert location < hint
    assert plan == -1 or hint < plan
