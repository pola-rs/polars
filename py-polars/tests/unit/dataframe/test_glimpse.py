from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any

import pytest

import polars as pl

TEST_DF = pl.DataFrame(
    {
        "a": [1.0, 2.8, 3.0],
        "b": [4, 5, None],
        "c": [True, False, True],
        "d": [None, "b", "c"],
        "e": ["usd", "eur", None],
        "f": pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            "1d",
            time_unit="us",
            eager=True,
        ),
        "g": pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            "1d",
            time_unit="ms",
            eager=True,
        ),
        "h": pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            "1d",
            time_unit="ns",
            eager=True,
        ),
        "i": [[5, 6], [3, 4], [9, 8]],
        "j": [[5.0, 6.0], [3.0, 4.0], [9.0, 8.0]],
        "k": [["A", "a"], ["B", "b"], ["C", "c"]],
    }
)

TEST_EXPECTED = textwrap.dedent(
    """\
    Rows: 3
    Columns: 11
    $ a          <f64> 1.0, 2.8, 3.0
    $ b          <i64> 4, 5, null
    $ c         <bool> True, False, True
    $ d          <str> null, 'b', 'c'
    $ e          <str> 'usd', 'eur', null
    $ f <datetime[μs]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
    $ g <datetime[ms]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
    $ h <datetime[ns]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
    $ i    <list[i64]> [5, 6], [3, 4], [9, 8]
    $ j    <list[f64]> [5.0, 6.0], [3.0, 4.0], [9.0, 8.0]
    $ k    <list[str]> ['A', 'a'], ['B', 'b'], ['C', 'c']
    """
)


def test_glimpse(capsys: Any) -> None:
    for result in (
        # check deprecated parameter still works
        TEST_DF.glimpse(return_as_string=True),  # type: ignore[call-overload]
        TEST_DF.glimpse(return_type="string"),
    ):
        assert result == TEST_EXPECTED


@pytest.mark.parametrize("return_type", [None, "self"])
def test_glimpse_print_return(return_type: str | None, capsys: Any) -> None:
    # default behaviour prints to stdout, returning nothing
    res = TEST_DF.glimpse(return_type=return_type)  # type: ignore[arg-type]

    if return_type is None:
        assert res is None
    else:
        assert res is TEST_DF

    # note: remove the last newline on the capsys
    assert capsys.readouterr().out[:-1] == TEST_EXPECTED


def test_glimpse_as_frame() -> None:
    result = TEST_DF.glimpse(return_type="frame")

    assert isinstance(result, pl.DataFrame)
    assert result.schema == pl.Schema(
        {
            "column": pl.String(),
            "dtype": pl.String(),
            "values": pl.List(pl.String),
        }
    )
    assert result.to_dict(as_series=False) == {
        "column": [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
        ],
        "dtype": [
            "f64",
            "i64",
            "bool",
            "str",
            "str",
            "datetime[μs]",
            "datetime[ms]",
            "datetime[ns]",
            "list[i64]",
            "list[f64]",
            "list[str]",
        ],
        "values": [
            ["1.0", "2.8", "3.0"],
            ["4", "5", None],
            ["True", "False", "True"],
            [None, "'b'", "'c'"],
            ["'usd'", "'eur'", None],
            ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00"],
            ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00"],
            ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00"],
            ["[5, 6]", "[3, 4]", "[9, 8]"],
            ["[5.0, 6.0]", "[3.0, 4.0]", "[9.0, 8.0]"],
            ["['A', 'a']", "['B', 'b']", "['C', 'c']"],
        ],
    }


def test_glimpse_colname_length() -> None:
    df = pl.DataFrame({"a" * 30: [11, 22, 33, 44, 55, 66]})
    result = df.glimpse(max_colname_length=20, return_type="string")

    expected = textwrap.dedent(
        """\
        Rows: 6
        Columns: 1
        $ aaaaaaaaaaaaaaaaaaa… <i64> 11, 22, 33, 44, 55, 66
        """
    )
    assert result == expected


def test_glimpse_items_length() -> None:
    df = pl.DataFrame({"n": range(50)}, schema={"n": pl.UInt8})

    # default max_items is 10
    result = df.glimpse(return_type="string")
    expected = textwrap.dedent(
        """\
        Rows: 50
        Columns: 1
        $ n <u8> 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        """
    )
    assert result == expected

    # test with custom max_items
    result = df.glimpse(max_items_per_column=5, return_type="string")
    expected = textwrap.dedent(
        """\
        Rows: 50
        Columns: 1
        $ n <u8> 0, 1, 2, 3, 4
        """
    )
    assert result == expected
