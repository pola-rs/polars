import textwrap
from datetime import datetime
from typing import Any

import polars as pl


def test_glimpse(capsys: Any) -> None:
    df = pl.DataFrame(
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
    result = df.glimpse(return_as_string=True)

    expected = textwrap.dedent(
        """\
        Rows: 3
        Columns: 11
        $ a          <f64> 1.0, 2.8, 3.0
        $ b          <i64> 4, 5, None
        $ c         <bool> True, False, True
        $ d          <str> None, 'b', 'c'
        $ e          <str> 'usd', 'eur', None
        $ f <datetime[μs]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
        $ g <datetime[ms]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
        $ h <datetime[ns]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
        $ i    <list[i64]> [5, 6], [3, 4], [9, 8]
        $ j    <list[f64]> [5.0, 6.0], [3.0, 4.0], [9.0, 8.0]
        $ k    <list[str]> ['A', 'a'], ['B', 'b'], ['C', 'c']
        """
    )
    assert result == expected

    # the default is to print to the console
    df.glimpse()
    # remove the last newline on the capsys
    assert capsys.readouterr().out[:-1] == expected

    colc = "a" * 96
    df = pl.DataFrame({colc: [11, 22, 33, 44, 55, 66]})
    result = df.glimpse(
        return_as_string=True, max_colname_length=20, max_items_per_column=4
    )
    expected = textwrap.dedent(
        """\
        Rows: 6
        Columns: 1
        $ aaaaaaaaaaaaaaaaaaa… <i64> 11, 22, 33, 44
        """
    )
    assert result == expected


def test_glimpse_colname_length() -> None:
    df = pl.DataFrame({"a" * 100: [1, 2, 3]})
    result = df.glimpse(max_colname_length=96, return_as_string=True)

    expected = f"$ {'a' * 95}… <i64> 1, 2, 3"
    assert result.strip().split("\n")[-1] == expected
