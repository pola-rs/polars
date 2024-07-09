from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_sql_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["xyz", "abcde", None]})
    sql_exprs = pl.sql_expr(
        [
            "MIN(a)",
            "POWER(a,a) AS aa",
            "SUBSTR(b,2,2) AS b2",
        ]
    )
    result = df.select(*sql_exprs)
    expected = pl.DataFrame(
        {"a": [1, 1, 1], "aa": [1, 4, 27], "b2": ["yz", "bc", None]}
    )
    assert_frame_equal(result, expected)

    # expect expressions that can't reasonably be parsed as expressions to raise
    # (for example: those that explicitly reference tables and/or use wildcards)
    with pytest.raises(
        SQLInterfaceError,
        match=r"unable to parse 'xyz\.\*' as Expr",
    ):
        pl.sql_expr("xyz.*")
