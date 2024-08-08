from __future__ import annotations

from datetime import timedelta

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


def test_bit_hex_literals() -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        out = ctx.execute(
            """
            SELECT *,
              -- bit strings
              b''                 AS b0,
              b'1001'             AS b1,
              b'11101011'         AS b2,
              b'1111110100110010' AS b3,
              -- hex strings
              x''                 AS x0,
              x'FF'               AS x1,
              x'4142'             AS x2,
              x'DeadBeef'         AS x3,
            FROM df
            """
        )

    assert out.to_dict(as_series=False) == {
        "b0": [b""],
        "b1": [b"\t"],
        "b2": [b"\xeb"],
        "b3": [b"\xfd2"],
        "x0": [b""],
        "x1": [b"\xff"],
        "x2": [b"AB"],
        "x3": [b"\xde\xad\xbe\xef"],
    }


def test_bit_hex_filter() -> None:
    df = pl.DataFrame(
        {"bin": [b"\x01", b"\x02", b"\x03", b"\x04"], "val": [9, 8, 7, 6]}
    )
    with pl.SQLContext(test=df) as ctx:
        for two in ("b'10'", "x'02'", "'\x02'", "b'0010'"):
            out = ctx.execute(f"SELECT val FROM test WHERE bin > {two}", eager=True)
            assert out.to_series().to_list() == [7, 6]


def test_bit_hex_errors() -> None:
    with pl.SQLContext(test=None) as ctx:
        with pytest.raises(
            SQLSyntaxError,
            match="bit string literal should contain only 0s and 1s",
        ):
            ctx.execute("SELECT b'007' FROM test", eager=True)

        with pytest.raises(
            SQLSyntaxError,
            match="hex string literal must have an even number of digits",
        ):
            ctx.execute("SELECT x'00F' FROM test", eager=True)

        with pytest.raises(
            SQLSyntaxError,
            match="hex string literal must have an even number of digits",
        ):
            pl.sql_expr("colx IN (x'FF',x'123')")

        with pytest.raises(
            SQLInterfaceError,
            match=r'NationalStringLiteral\("hmmm"\) is not a supported literal',
        ):
            pl.sql_expr("N'hmmm'")


def test_bit_hex_membership() -> None:
    df = pl.DataFrame(
        {
            "x": [b"\x05", b"\xff", b"\xcc", b"\x0b"],
            "y": [1, 2, 3, 4],
        }
    )
    # this checks the internal `visit_any_value` codepath
    for values in (
        "b'0101', b'1011'",
        "x'05', x'0b'",
    ):
        dff = df.filter(pl.sql_expr(f"x IN ({values})"))
        assert dff["y"].to_list() == [1, 4]


def test_dollar_quoted_literals() -> None:
    df = pl.sql(
        """
        SELECT
          $$xyz$$ AS dq1,
          $q$xyz$q$ AS dq2,
          $tag$xyz$tag$ AS dq3,
          $QUOTE$xyz$QUOTE$ AS dq4,
        """
    ).collect()
    assert df.to_dict(as_series=False) == {f"dq{n}": ["xyz"] for n in range(1, 5)}

    df = pl.sql("SELECT $$x$z$$ AS dq").collect()
    assert df.item() == "x$z"


def test_intervals() -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        out = ctx.execute(
            """
            SELECT
              -- short form with/without spaces
              INTERVAL '1w2h3m4s' AS i1,
              INTERVAL '100ms 100us' AS i2,
              -- long form with/without commas (case-insensitive)
              INTERVAL '1 week, 2 hours, 3 minutes, 4 seconds' AS i3,
              INTERVAL '1 QUARTER 2 Months 987 microseconds' AS i4,
            FROM df
            """
        )
        expected = pl.DataFrame(
            {
                "i1": [timedelta(weeks=1, hours=2, minutes=3, seconds=4)],
                "i2": [timedelta(microseconds=100100)],
                "i3": [timedelta(weeks=1, hours=2, minutes=3, seconds=4)],
                "i4": [timedelta(days=140, microseconds=987)],
            },
        ).cast(pl.Duration("ns"))

        assert_frame_equal(expected, out)

        # TODO: negative intervals
        with pytest.raises(
            SQLInterfaceError,
            match="minus signs are not yet supported in interval strings; found '-7d'",
        ):
            ctx.execute("SELECT INTERVAL '-7d' AS one_week_ago FROM df")

        with pytest.raises(
            SQLSyntaxError,
            match="unary ops are not valid on interval strings; found -'7d'",
        ):
            ctx.execute("SELECT INTERVAL -'7d' AS one_week_ago FROM df")


def test_select_literals_no_table() -> None:
    res = pl.sql("SELECT 1 AS one, '2' AS two, 3.0 AS three", eager=True)
    assert res.to_dict(as_series=False) == {
        "one": [1],
        "two": ["2"],
        "three": [3.0],
    }


def test_select_from_table_with_reserved_names() -> None:
    select = pl.DataFrame({"select": [1, 2, 3], "from": [4, 5, 6]})  # noqa: F841
    out = pl.sql(
        """
        SELECT "from", "select"
          FROM "select"
          WHERE "from" >= 5 AND "select" % 2 != 1
        """,
        eager=True,
    )
    assert out.rows() == [(5, 2)]
