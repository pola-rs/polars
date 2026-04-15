from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
import polars._plr as plr
from polars._utils.unstable import issue_unstable_warning
from polars.config import _POLARS_CFG_ENV_VARS

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _environ() -> Iterator[None]:
    """Fixture to restore the environment after/during tests."""
    with pl.Config(restore_defaults=True):
        yield


def test_ascii_tables() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [4, 5],
            "c": [[list(range(1, 26))], [list(range(1, 76))]],
        }
    )

    ascii_table_repr = (
        "shape: (2, 3)\n"
        "+-----+-----+------------------+\n"
        "| a   | b   | c                |\n"
        "| --- | --- | ---              |\n"
        "| i64 | i64 | list[list[i64]]  |\n"
        "+==============================+\n"
        "| 1   | 4   | [[1, 2, ... 25]] |\n"
        "| 2   | 5   | [[1, 2, ... 75]] |\n"
        "+-----+-----+------------------+"
    )
    # note: expect to render ascii only within the given scope
    with pl.Config(set_ascii_tables=True):
        assert repr(df) == ascii_table_repr

    # confirm back to utf8 default after scope-exit
    assert (
        repr(df) == "shape: (2, 3)\n"
        "┌─────┬─────┬─────────────────┐\n"
        "│ a   ┆ b   ┆ c               │\n"
        "│ --- ┆ --- ┆ ---             │\n"
        "│ i64 ┆ i64 ┆ list[list[i64]] │\n"
        "╞═════╪═════╪═════════════════╡\n"
        "│ 1   ┆ 4   ┆ [[1, 2, … 25]]  │\n"
        "│ 2   ┆ 5   ┆ [[1, 2, … 75]]  │\n"
        "└─────┴─────┴─────────────────┘"
    )

    @pl.Config(set_ascii_tables=True)
    def ascii_table() -> str:
        return repr(df)

    assert ascii_table() == ascii_table_repr


def test_hide_header_elements() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    pl.Config.set_tbl_hide_column_data_types(True)
    assert (
        str(df) == "shape: (3, 3)\n"
        "┌───┬───┬───┐\n"
        "│ a ┆ b ┆ c │\n"
        "╞═══╪═══╪═══╡\n"
        "│ 1 ┆ 4 ┆ 7 │\n"
        "│ 2 ┆ 5 ┆ 8 │\n"
        "│ 3 ┆ 6 ┆ 9 │\n"
        "└───┴───┴───┘"
    )

    pl.Config.set_tbl_hide_column_data_types(False).set_tbl_hide_column_names(True)
    assert (
        str(df) == "shape: (3, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 4   ┆ 7   │\n"
        "│ 2   ┆ 5   ┆ 8   │\n"
        "│ 3   ┆ 6   ┆ 9   │\n"
        "└─────┴─────┴─────┘"
    )


def test_set_tbl_cols() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    pl.Config.set_tbl_cols(1)
    assert str(df).split("\n")[2] == "│ a   ┆ … │"
    pl.Config.set_tbl_cols(2)
    assert str(df).split("\n")[2] == "│ a   ┆ … ┆ c   │"
    pl.Config.set_tbl_cols(3)
    assert str(df).split("\n")[2] == "│ a   ┆ b   ┆ c   │"

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
    )
    pl.Config.set_tbl_cols(2)
    assert str(df).split("\n")[2] == "│ a   ┆ … ┆ d   │"
    pl.Config.set_tbl_cols(3)
    assert str(df).split("\n")[2] == "│ a   ┆ b   ┆ … ┆ d   │"
    pl.Config.set_tbl_cols(-1)
    assert str(df).split("\n")[2] == "│ a   ┆ b   ┆ c   ┆ d   │"


def test_set_tbl_rows() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]})
    ser = pl.Series("ser", [1, 2, 3, 4, 5])

    pl.Config.set_tbl_rows(0)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ …   ┆ …   ┆ …   │\n"
        "└─────┴─────┴─────┘"
    )
    assert str(ser) == "shape: (5,)\nSeries: 'ser' [i64]\n[\n\t…\n]"

    pl.Config.set_tbl_rows(1)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "│ …   ┆ …   ┆ …   │\n"
        "└─────┴─────┴─────┘"
    )
    assert str(ser) == "shape: (5,)\nSeries: 'ser' [i64]\n[\n\t1\n\t…\n]"

    pl.Config.set_tbl_rows(2)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "│ …   ┆ …   ┆ …   │\n"
        "│ 4   ┆ 8   ┆ 12  │\n"
        "└─────┴─────┴─────┘"
    )
    assert str(ser) == "shape: (5,)\nSeries: 'ser' [i64]\n[\n\t1\n\t…\n\t5\n]"

    pl.Config.set_tbl_rows(3)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "│ 2   ┆ 6   ┆ 10  │\n"
        "│ …   ┆ …   ┆ …   │\n"
        "│ 4   ┆ 8   ┆ 12  │\n"
        "└─────┴─────┴─────┘"
    )
    assert str(ser) == "shape: (5,)\nSeries: 'ser' [i64]\n[\n\t1\n\t2\n\t…\n\t5\n]"

    pl.Config.set_tbl_rows(4)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "│ 2   ┆ 6   ┆ 10  │\n"
        "│ 3   ┆ 7   ┆ 11  │\n"
        "│ 4   ┆ 8   ┆ 12  │\n"
        "└─────┴─────┴─────┘"
    )
    assert str(ser) == "shape: (5,)\nSeries: 'ser' [i64]\n[\n\t1\n\t2\n\t…\n\t4\n\t5\n]"

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [6, 7, 8, 9, 10],
            "c": [11, 12, 13, 14, 15],
        }
    )

    pl.Config.set_tbl_rows(3)
    assert (
        str(df) == "shape: (5, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 6   ┆ 11  │\n"
        "│ 2   ┆ 7   ┆ 12  │\n"
        "│ …   ┆ …   ┆ …   │\n"
        "│ 5   ┆ 10  ┆ 15  │\n"
        "└─────┴─────┴─────┘"
    )

    pl.Config.set_tbl_rows(-1)
    assert str(ser) == "shape: (5,)\nSeries: 'ser' [i64]\n[\n\t1\n\t2\n\t3\n\t4\n\t5\n]"

    pl.Config.set_tbl_hide_dtype_separator(True)
    assert (
        str(df) == "shape: (5, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 6   ┆ 11  │\n"
        "│ 2   ┆ 7   ┆ 12  │\n"
        "│ 3   ┆ 8   ┆ 13  │\n"
        "│ 4   ┆ 9   ┆ 14  │\n"
        "│ 5   ┆ 10  ┆ 15  │\n"
        "└─────┴─────┴─────┘"
    )


def test_set_tbl_formats() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    pl.Config().set_tbl_formatting("ASCII_MARKDOWN")
    assert str(df) == (
        "shape: (3, 3)\n"
        "| foo | bar | ham |\n"
        "| --- | --- | --- |\n"
        "| i64 | f64 | str |\n"
        "|-----|-----|-----|\n"
        "| 1   | 6.0 | a   |\n"
        "| 2   | 7.0 | b   |\n"
        "| 3   | 8.0 | c   |"
    )

    pl.Config().set_tbl_formatting("ASCII_BORDERS_ONLY_CONDENSED")
    with pl.Config(tbl_hide_dtype_separator=True):
        assert str(df) == (
            "shape: (3, 3)\n"
            "+-----------------+\n"
            "| foo   bar   ham |\n"
            "| i64   f64   str |\n"
            "+=================+\n"
            "| 1     6.0   a   |\n"
            "| 2     7.0   b   |\n"
            "| 3     8.0   c   |\n"
            "+-----------------+"
        )

    # temporarily scope "nothing" style, with no data types
    with pl.Config(
        tbl_formatting="NOTHING",
        tbl_hide_column_data_types=True,
    ):
        assert str(df) == (
            "shape: (3, 3)\n"
            " foo  bar  ham \n"
            " 1    6.0  a   \n"
            " 2    7.0  b   \n"
            " 3    8.0  c   "
        )

    # after scope, expect previous style
    assert str(df) == (
        "shape: (3, 3)\n"
        "+-----------------+\n"
        "| foo   bar   ham |\n"
        "| ---   ---   --- |\n"
        "| i64   f64   str |\n"
        "+=================+\n"
        "| 1     6.0   a   |\n"
        "| 2     7.0   b   |\n"
        "| 3     8.0   c   |\n"
        "+-----------------+"
    )

    # invalid style
    with pytest.raises(ValueError, match="invalid table format name: 'NOPE'"):
        pl.Config().set_tbl_formatting("NOPE")  # type: ignore[arg-type]


def test_set_tbl_width_chars() -> None:
    df = pl.DataFrame(
        {
            "a really long col": [1, 2, 3],
            "b": ["", "this is a string value that will be truncated", None],
            "this is 10": [4, 5, 6],
        }
    )
    assert max(len(line) for line in str(df).split("\n")) == 68

    pl.Config.set_tbl_width_chars(60)
    assert max(len(line) for line in str(df).split("\n")) == 60

    # force minimal table size (will hard-wrap everything; "don't try this at home" :p)
    pl.Config.set_tbl_width_chars(0)
    assert max(len(line) for line in str(df).split("\n")) == 19

    # this check helps to check that column width bucketing
    # is exact; no extraneous character allocation
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        },
        schema_overrides={"A": pl.Int64, "B": pl.Int64},
    ).select(pl.all(), pl.all().name.suffix("_suffix!"))

    with pl.Config(tbl_width_chars=87):
        assert max(len(line) for line in str(df).split("\n")) == 87

    # check that -1 is interpreted as no limit
    df = pl.DataFrame({str(i): ["a" * 25] for i in range(5)})
    for tbl_width_chars, expected_width in [
        (None, 100),
        (-1, 141),
    ]:
        with pl.Config(tbl_width_chars=tbl_width_chars):
            assert max(len(line) for line in str(df).split("\n")) == expected_width


def test_shape_below_table_and_inlined_dtype() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    pl.Config.set_tbl_column_data_type_inline(True).set_tbl_dataframe_shape_below(True)
    pl.Config.set_tbl_formatting("UTF8_FULL", rounded_corners=True)
    assert (
        str(df) == ""
        "╭─────────┬─────────┬─────────╮\n"
        "│ a (i64) ┆ b (i64) ┆ c (i64) │\n"
        "╞═════════╪═════════╪═════════╡\n"
        "│ 1       ┆ 3       ┆ 5       │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 2       ┆ 4       ┆ 6       │\n"
        "╰─────────┴─────────┴─────────╯\n"
        "shape: (2, 3)"
    )

    pl.Config.set_tbl_dataframe_shape_below(False)
    assert (
        str(df) == "shape: (2, 3)\n"
        "╭─────────┬─────────┬─────────╮\n"
        "│ a (i64) ┆ b (i64) ┆ c (i64) │\n"
        "╞═════════╪═════════╪═════════╡\n"
        "│ 1       ┆ 3       ┆ 5       │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 2       ┆ 4       ┆ 6       │\n"
        "╰─────────┴─────────┴─────────╯"
    )
    (
        pl.Config.set_tbl_formatting(None, rounded_corners=False)
        .set_tbl_column_data_type_inline(False)
        .set_tbl_cell_alignment("right")
    )
    assert (
        str(df) == "shape: (2, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│   a ┆   b ┆   c │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│   1 ┆   3 ┆   5 │\n"
        "│   2 ┆   4 ┆   6 │\n"
        "└─────┴─────┴─────┘"
    )
    with pytest.raises(ValueError):
        pl.Config.set_tbl_cell_alignment("INVALID")  # type: ignore[arg-type]


def test_shape_format_for_big_numbers() -> None:
    df = pl.DataFrame({"a": range(1, 1001), "b": range(1001, 1001 + 1000)})

    pl.Config.set_tbl_column_data_type_inline(True).set_tbl_dataframe_shape_below(True)
    pl.Config.set_tbl_formatting("UTF8_FULL", rounded_corners=True)
    assert (
        str(df) == ""
        "╭─────────┬─────────╮\n"
        "│ a (i64) ┆ b (i64) │\n"
        "╞═════════╪═════════╡\n"
        "│ 1       ┆ 1001    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 2       ┆ 1002    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 3       ┆ 1003    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 4       ┆ 1004    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 5       ┆ 1005    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ …       ┆ …       │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 996     ┆ 1996    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 997     ┆ 1997    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 998     ┆ 1998    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 999     ┆ 1999    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 1000    ┆ 2000    │\n"
        "╰─────────┴─────────╯\n"
        "shape: (1_000, 2)"
    )

    pl.Config.set_tbl_column_data_type_inline(True).set_tbl_dataframe_shape_below(False)
    assert (
        str(df) == "shape: (1_000, 2)\n"
        "╭─────────┬─────────╮\n"
        "│ a (i64) ┆ b (i64) │\n"
        "╞═════════╪═════════╡\n"
        "│ 1       ┆ 1001    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 2       ┆ 1002    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 3       ┆ 1003    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 4       ┆ 1004    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 5       ┆ 1005    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ …       ┆ …       │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 996     ┆ 1996    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 997     ┆ 1997    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 998     ┆ 1998    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 999     ┆ 1999    │\n"
        "├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤\n"
        "│ 1000    ┆ 2000    │\n"
        "╰─────────┴─────────╯"
    )

    pl.Config.set_tbl_rows(0)
    ser = pl.Series("ser", range(1000))
    assert str(ser) == "shape: (1_000,)\nSeries: 'ser' [i64]\n[\n\t…\n]"

    pl.Config.set_tbl_rows(1)
    pl.Config.set_tbl_cols(1)
    df = pl.DataFrame({str(col_num): 1 for col_num in range(1000)})

    assert (
        str(df) == "shape: (1, 1_000)\n"
        "╭─────────┬───╮\n"
        "│ 0 (i64) ┆ … │\n"
        "╞═════════╪═══╡\n"
        "│ 1       ┆ … │\n"
        "╰─────────┴───╯"
    )

    pl.Config.set_tbl_formatting("ASCII_FULL_CONDENSED")
    assert (
        str(df) == "shape: (1, 1_000)\n"
        "+---------+-----+\n"
        "| 0 (i64) | ... |\n"
        "+===============+\n"
        "| 1       | ... |\n"
        "+---------+-----+"
    )


def test_numeric_right_alignment() -> None:
    pl.Config.set_tbl_cell_numeric_alignment("RIGHT")

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert (
        str(df) == "shape: (3, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│   a ┆   b ┆   c │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│   1 ┆   4 ┆   7 │\n"
        "│   2 ┆   5 ┆   8 │\n"
        "│   3 ┆   6 ┆   9 │\n"
        "└─────┴─────┴─────┘"
    )

    df = pl.DataFrame(
        {"a": [1.1, 2.22, 3.333], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
    )
    with pl.Config():
        pl.Config.set_fmt_float("full")
        assert (
            str(df) == "shape: (3, 3)\n"
            "┌───────┬─────┬─────┐\n"
            "│     a ┆   b ┆   c │\n"
            "│   --- ┆ --- ┆ --- │\n"
            "│   f64 ┆ f64 ┆ f64 │\n"
            "╞═══════╪═════╪═════╡\n"
            "│   1.1 ┆   4 ┆   7 │\n"
            "│  2.22 ┆   5 ┆   8 │\n"
            "│ 3.333 ┆   6 ┆   9 │\n"
            "└───────┴─────┴─────┘"
        )

    with pl.Config(fmt_float="mixed"):
        assert (
            str(df) == "shape: (3, 3)\n"
            "┌───────┬─────┬─────┐\n"
            "│     a ┆   b ┆   c │\n"
            "│   --- ┆ --- ┆ --- │\n"
            "│   f64 ┆ f64 ┆ f64 │\n"
            "╞═══════╪═════╪═════╡\n"
            "│   1.1 ┆ 4.0 ┆ 7.0 │\n"
            "│  2.22 ┆ 5.0 ┆ 8.0 │\n"
            "│ 3.333 ┆ 6.0 ┆ 9.0 │\n"
            "└───────┴─────┴─────┘"
        )

    with pl.Config(float_precision=6):
        assert str(df) == (
            "shape: (3, 3)\n"
            "┌──────────┬──────────┬──────────┐\n"
            "│        a ┆        b ┆        c │\n"
            "│      --- ┆      --- ┆      --- │\n"
            "│      f64 ┆      f64 ┆      f64 │\n"
            "╞══════════╪══════════╪══════════╡\n"
            "│ 1.100000 ┆ 4.000000 ┆ 7.000000 │\n"
            "│ 2.220000 ┆ 5.000000 ┆ 8.000000 │\n"
            "│ 3.333000 ┆ 6.000000 ┆ 9.000000 │\n"
            "└──────────┴──────────┴──────────┘"
        )
        with pl.Config(float_precision=None):
            assert (
                str(df) == "shape: (3, 3)\n"
                "┌───────┬─────┬─────┐\n"
                "│     a ┆   b ┆   c │\n"
                "│   --- ┆ --- ┆ --- │\n"
                "│   f64 ┆ f64 ┆ f64 │\n"
                "╞═══════╪═════╪═════╡\n"
                "│   1.1 ┆ 4.0 ┆ 7.0 │\n"
                "│  2.22 ┆ 5.0 ┆ 8.0 │\n"
                "│ 3.333 ┆ 6.0 ┆ 9.0 │\n"
                "└───────┴─────┴─────┘"
            )

    df = pl.DataFrame(
        {"a": [1.1, 22.2, 3.33], "b": [444.0, 55.5, 6.6], "c": [77.7, 8888.0, 9.9999]}
    )
    with pl.Config(fmt_float="full", float_precision=1):
        assert (
            str(df) == "shape: (3, 3)\n"
            "┌──────┬───────┬────────┐\n"
            "│    a ┆     b ┆      c │\n"
            "│  --- ┆   --- ┆    --- │\n"
            "│  f64 ┆   f64 ┆    f64 │\n"
            "╞══════╪═══════╪════════╡\n"
            "│  1.1 ┆ 444.0 ┆   77.7 │\n"
            "│ 22.2 ┆  55.5 ┆ 8888.0 │\n"
            "│  3.3 ┆   6.6 ┆   10.0 │\n"
            "└──────┴───────┴────────┘"
        )

    df = pl.DataFrame(
        {
            "a": [1100000000000000000.1, 22200000000000000.2, 33330000000000000.33333],
            "b": [40000000000000000000.0, 5, 600000000000000000.0],
            "c": [700000.0, 80000000000000000.0, 900],
        }
    )
    with pl.Config(float_precision=2):
        assert (
            str(df) == "shape: (3, 3)\n"
            "┌─────────┬─────────┬───────────┐\n"
            "│       a ┆       b ┆         c │\n"
            "│     --- ┆     --- ┆       --- │\n"
            "│     f64 ┆     f64 ┆       f64 │\n"
            "╞═════════╪═════════╪═══════════╡\n"
            "│ 1.10e18 ┆ 4.00e19 ┆ 700000.00 │\n"
            "│ 2.22e16 ┆    5.00 ┆   8.00e16 │\n"
            "│ 3.33e16 ┆ 6.00e17 ┆    900.00 │\n"
            "└─────────┴─────────┴───────────┘"
        )


@pytest.mark.write_disk
def test_config_load_save(tmp_path: Path) -> None:
    for file in (
        None,
        tmp_path / "polars.config",
        str(tmp_path / "polars.config"),
    ):
        # set some config options...
        pl.Config.set_tbl_cols(12)
        pl.Config.set_verbose(True)
        pl.Config.set_fmt_float("full")
        pl.Config.set_float_precision(6)
        pl.Config.set_thousands_separator(",")
        assert os.environ.get("POLARS_VERBOSE") == "1"

        if file is None:
            cfg = pl.Config.save()
            assert isinstance(cfg, str)
        else:
            assert pl.Config.save_to_file(file) is None

        assert "POLARS_VERBOSE" in pl.Config.state(if_set=True)

        # ...modify the same options...
        pl.Config.set_tbl_cols(10)
        pl.Config.set_verbose(False)
        pl.Config.set_fmt_float("mixed")
        pl.Config.set_float_precision(2)
        pl.Config.set_thousands_separator(None)
        assert os.environ.get("POLARS_VERBOSE") == "0"

        # ...load back from config file/string...
        assert isinstance(cfg, str)
        if file is None:
            pl.Config.load(cfg)
        else:
            with pytest.raises(ValueError, match="invalid Config file"):
                pl.Config.load_from_file(cfg)

            if isinstance(file, Path):
                with pytest.raises(TypeError, match="the JSON object must be str"):
                    pl.Config.load(file)  # type: ignore[arg-type]
            else:
                with pytest.raises(ValueError, match="invalid Config string"):
                    pl.Config.load(file)

            pl.Config.load_from_file(file)

        # ...and confirm the saved options were set.
        assert os.environ.get("POLARS_FMT_MAX_COLS") == "12"
        assert os.environ.get("POLARS_VERBOSE") == "1"
        assert plr.get_float_fmt() == "full"
        assert plr.get_float_precision() == 6

        # restore all default options (unsets from env)
        pl.Config.restore_defaults()
        for e in ("POLARS_FMT_MAX_COLS", "POLARS_VERBOSE"):
            assert e not in pl.Config.state(if_set=True)
            assert e in pl.Config.state()

        assert os.environ.get("POLARS_FMT_MAX_COLS") is None
        assert os.environ.get("POLARS_VERBOSE") is None
        assert plr.get_float_fmt() == "mixed"
        assert plr.get_float_precision() is None

    # ref: #11094
    with pl.Config(
        streaming_chunk_size=100,
        tbl_cols=2000,
        tbl_formatting="UTF8_NO_BORDERS",
        tbl_hide_column_data_types=True,
        tbl_hide_dtype_separator=True,
        tbl_rows=2000,
        tbl_width_chars=2000,
        verbose=True,
    ):
        assert isinstance(repr(pl.DataFrame({"xyz": [0]})), str)


def test_config_load_save_context() -> None:
    # Store the default configuration state.
    default_state = pl.Config.save()

    # Establish some non-default settings.
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_verbose(True)

    # Load the default config, validate load & context manager behaviour.
    with pl.Config.load(default_state):
        assert os.environ.get("POLARS_FMT_TABLE_FORMATTING") is None
        assert os.environ.get("POLARS_VERBOSE") is None

    # Ensure earlier state was restored.
    assert os.environ["POLARS_FMT_TABLE_FORMATTING"] == "ASCII_MARKDOWN"
    assert os.environ["POLARS_VERBOSE"]

    # Restore defaults for other tests.
    pl.Config.restore_defaults()


def test_config_instances() -> None:
    # establish two config instances that defer setting their options
    cfg_markdown = pl.Config(
        tbl_formatting="MARKDOWN",
        apply_on_context_enter=True,
    )
    cfg_compact = pl.Config(
        tbl_rows=4,
        tbl_cols=4,
        tbl_column_data_type_inline=True,
        apply_on_context_enter=True,
    )

    # check instance (in)equality
    assert cfg_markdown != cfg_compact
    assert cfg_markdown == pl.Config(
        tbl_formatting="MARKDOWN", apply_on_context_enter=True
    )

    # confirm that the options have not been applied yet
    assert os.environ.get("POLARS_FMT_TABLE_FORMATTING") is None

    # confirm that the deferred options are applied when the instance context
    # is entered into, and that they can be re-used without leaking state
    @cfg_markdown
    def fn1() -> str | None:
        return os.environ.get("POLARS_FMT_TABLE_FORMATTING")

    assert fn1() == "MARKDOWN"
    assert os.environ.get("POLARS_FMT_TABLE_FORMATTING") is None

    with cfg_markdown:  # can re-use instance as decorator and context
        assert os.environ.get("POLARS_FMT_TABLE_FORMATTING") == "MARKDOWN"
    assert os.environ.get("POLARS_FMT_TABLE_FORMATTING") is None

    @cfg_markdown
    def fn2() -> str | None:
        return os.environ.get("POLARS_FMT_TABLE_FORMATTING")

    assert fn2() == "MARKDOWN"
    assert os.environ.get("POLARS_FMT_TABLE_FORMATTING") is None

    df = pl.DataFrame({f"c{idx}": [idx] * 10 for idx in range(10)})

    @cfg_compact
    def fn3(df: pl.DataFrame) -> str:
        return repr(df)

    # reuse config instance and confirm state does not leak between invocations
    for _ in range(3):
        assert (
            fn3(df)
            == dedent("""
            shape: (10, 10)
            ┌──────────┬──────────┬───┬──────────┬──────────┐
            │ c0 (i64) ┆ c1 (i64) ┆ … ┆ c8 (i64) ┆ c9 (i64) │
            ╞══════════╪══════════╪═══╪══════════╪══════════╡
            │ 0        ┆ 1        ┆ … ┆ 8        ┆ 9        │
            │ 0        ┆ 1        ┆ … ┆ 8        ┆ 9        │
            │ …        ┆ …        ┆ … ┆ …        ┆ …        │
            │ 0        ┆ 1        ┆ … ┆ 8        ┆ 9        │
            │ 0        ┆ 1        ┆ … ┆ 8        ┆ 9        │
            └──────────┴──────────┴───┴──────────┴──────────┘""").lstrip()
        )

        assert (
            repr(df)
            == dedent("""
            shape: (10, 10)
            ┌─────┬─────┬─────┬─────┬───┬─────┬─────┬─────┬─────┐
            │ c0  ┆ c1  ┆ c2  ┆ c3  ┆ … ┆ c6  ┆ c7  ┆ c8  ┆ c9  │
            │ --- ┆ --- ┆ --- ┆ --- ┆   ┆ --- ┆ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 ┆ i64 ┆   ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╪═════╪═══╪═════╪═════╪═════╪═════╡
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            │ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 6   ┆ 7   ┆ 8   ┆ 9   │
            └─────┴─────┴─────┴─────┴───┴─────┴─────┴─────┴─────┘""").lstrip()
        )


def test_config_scope() -> None:
    pl.Config.set_verbose(False)
    pl.Config.set_tbl_cols(8)

    initial_state = pl.Config.state()

    with pl.Config() as cfg:
        (
            cfg.set_tbl_formatting(rounded_corners=True)
            .set_verbose(True)
            .set_tbl_hide_dtype_separator(True)
            .set_ascii_tables()
        )
        new_state_entries = set(
            {
                "POLARS_FMT_MAX_COLS": "8",
                "POLARS_FMT_TABLE_FORMATTING": "ASCII_FULL_CONDENSED",
                "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR": "1",
                "POLARS_FMT_TABLE_ROUNDED_CORNERS": "1",
                "POLARS_VERBOSE": "1",
            }.items()
        )
        assert set(initial_state.items()) != new_state_entries
        assert new_state_entries.issubset(set(cfg.state().items()))

    # expect scope-exit to restore original state
    assert pl.Config.state() == initial_state


def test_config_raise_error_if_not_exist() -> None:
    with pytest.raises(AttributeError), pl.Config(i_do_not_exist=True):  # type: ignore[call-arg]
        pass


def test_config_state_env_only() -> None:
    with pl.Config() as cfg:
        cfg.set_verbose(False)
        cfg.set_fmt_float("full")

        state_all = cfg.state(env_only=False)
        state_env_only = cfg.state(env_only=True)
        assert len(state_env_only) < len(state_all)
        assert "set_fmt_float" in state_all
        assert "set_fmt_float" not in state_env_only


def test_set_streaming_chunk_size() -> None:
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(8)
        assert os.environ.get("POLARS_STREAMING_CHUNK_SIZE") == "8"

    with pytest.raises(ValueError), pl.Config() as cfg:
        cfg.set_streaming_chunk_size(0)


def test_set_fmt_str_lengths_invalid_length() -> None:
    with pl.Config() as cfg:
        with pytest.raises(ValueError):
            cfg.set_fmt_str_lengths(0)
        with pytest.raises(ValueError):
            cfg.set_fmt_str_lengths(-2)


def test_truncated_rows_cols_values_ascii() -> None:
    df = pl.DataFrame({f"c{n}": list(range(-n, 100 - n)) for n in range(10)})

    pl.Config.set_tbl_formatting("UTF8_BORDERS_ONLY", rounded_corners=True)
    assert (
        str(df) == "shape: (100, 10)\n"
        "╭───────────────────────────────────────────────────╮\n"
        "│ c0    c1    c2    c3    …   c6    c7    c8    c9  │\n"
        "│ ---   ---   ---   ---       ---   ---   ---   --- │\n"
        "│ i64   i64   i64   i64       i64   i64   i64   i64 │\n"
        "╞═══════════════════════════════════════════════════╡\n"
        "│ 0     -1    -2    -3    …   -6    -7    -8    -9  │\n"
        "│ 1     0     -1    -2    …   -5    -6    -7    -8  │\n"
        "│ 2     1     0     -1    …   -4    -5    -6    -7  │\n"
        "│ 3     2     1     0     …   -3    -4    -5    -6  │\n"
        "│ 4     3     2     1     …   -2    -3    -4    -5  │\n"
        "│ …     …     …     …     …   …     …     …     …   │\n"
        "│ 95    94    93    92    …   89    88    87    86  │\n"
        "│ 96    95    94    93    …   90    89    88    87  │\n"
        "│ 97    96    95    94    …   91    90    89    88  │\n"
        "│ 98    97    96    95    …   92    91    90    89  │\n"
        "│ 99    98    97    96    …   93    92    91    90  │\n"
        "╰───────────────────────────────────────────────────╯"
    )
    with pl.Config(tbl_formatting="ASCII_FULL_CONDENSED"):
        assert (
            str(df) == "shape: (100, 10)\n"
            "+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n"
            "| c0  | c1  | c2  | c3  | ... | c6  | c7  | c8  | c9  |\n"
            "| --- | --- | --- | --- |     | --- | --- | --- | --- |\n"
            "| i64 | i64 | i64 | i64 |     | i64 | i64 | i64 | i64 |\n"
            "+=====================================================+\n"
            "| 0   | -1  | -2  | -3  | ... | -6  | -7  | -8  | -9  |\n"
            "| 1   | 0   | -1  | -2  | ... | -5  | -6  | -7  | -8  |\n"
            "| 2   | 1   | 0   | -1  | ... | -4  | -5  | -6  | -7  |\n"
            "| 3   | 2   | 1   | 0   | ... | -3  | -4  | -5  | -6  |\n"
            "| 4   | 3   | 2   | 1   | ... | -2  | -3  | -4  | -5  |\n"
            "| ... | ... | ... | ... | ... | ... | ... | ... | ... |\n"
            "| 95  | 94  | 93  | 92  | ... | 89  | 88  | 87  | 86  |\n"
            "| 96  | 95  | 94  | 93  | ... | 90  | 89  | 88  | 87  |\n"
            "| 97  | 96  | 95  | 94  | ... | 91  | 90  | 89  | 88  |\n"
            "| 98  | 97  | 96  | 95  | ... | 92  | 91  | 90  | 89  |\n"
            "| 99  | 98  | 97  | 96  | ... | 93  | 92  | 91  | 90  |\n"
            "+-----+-----+-----+-----+-----+-----+-----+-----+-----+"
        )

    with pl.Config(tbl_formatting="MARKDOWN"):
        df = pl.DataFrame({"b": [b"0tigohij1prisdfj1gs2io3fbjg0pfihodjgsnfbbmfgnd8j"]})
        assert (
            str(df)
            == dedent("""
            shape: (1, 1)
            | b                               |
            | ---                             |
            | binary                          |
            |---------------------------------|
            | b"0tigohij1prisdfj1gs2io3fbjg0… |""").lstrip()
        )

    with pl.Config(tbl_formatting="ASCII_MARKDOWN"):
        df = pl.DataFrame({"b": [b"0tigohij1prisdfj1gs2io3fbjg0pfihodjgsnfbbmfgnd8j"]})
        assert (
            str(df)
            == dedent("""
            shape: (1, 1)
            | b                                 |
            | ---                               |
            | binary                            |
            |-----------------------------------|
            | b"0tigohij1prisdfj1gs2io3fbjg0... |""").lstrip()
        )


def test_warn_unstable(recwarn: pytest.WarningsRecorder) -> None:
    issue_unstable_warning()
    assert len(recwarn) == 0

    pl.Config().warn_unstable(True)

    issue_unstable_warning()
    assert len(recwarn) == 1

    pl.Config().warn_unstable(False)

    issue_unstable_warning()
    assert len(recwarn) == 1


@pytest.mark.parametrize(
    ("environment_variable", "config_setting", "value", "expected"),
    [
        ("POLARS_ENGINE_AFFINITY", "set_engine_affinity", "gpu", "gpu"),
        ("POLARS_FMT_MAX_COLS", "set_tbl_cols", 12, "12"),
        ("POLARS_FMT_MAX_ROWS", "set_tbl_rows", 3, "3"),
        ("POLARS_FMT_STR_LEN", "set_fmt_str_lengths", 42, "42"),
        ("POLARS_FMT_TABLE_CELL_ALIGNMENT", "set_tbl_cell_alignment", "RIGHT", "RIGHT"),
        (
            "POLARS_FMT_TABLE_CELL_NUMERIC_ALIGNMENT",
            "set_tbl_cell_numeric_alignment",
            "RIGHT",
            "RIGHT",
        ),
        ("POLARS_FMT_TABLE_HIDE_COLUMN_NAMES", "set_tbl_hide_column_names", True, "1"),
        (
            "POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW",
            "set_tbl_dataframe_shape_below",
            True,
            "1",
        ),
        (
            "POLARS_FMT_TABLE_FORMATTING",
            "set_ascii_tables",
            True,
            "ASCII_FULL_CONDENSED",
        ),
        (
            "POLARS_FMT_TABLE_FORMATTING",
            "set_tbl_formatting",
            "ASCII_MARKDOWN",
            "ASCII_MARKDOWN",
        ),
        (
            "POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES",
            "set_tbl_hide_column_data_types",
            True,
            "1",
        ),
        (
            "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR",
            "set_tbl_hide_dtype_separator",
            True,
            "1",
        ),
        (
            "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION",
            "set_tbl_hide_dataframe_shape",
            True,
            "1",
        ),
        (
            "POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE",
            "set_tbl_column_data_type_inline",
            True,
            "1",
        ),
        ("POLARS_STREAMING_CHUNK_SIZE", "set_streaming_chunk_size", 100, "100"),
        ("POLARS_TABLE_WIDTH", "set_tbl_width_chars", 80, "80"),
        ("POLARS_VERBOSE", "set_verbose", True, "1"),
        ("POLARS_WARN_UNSTABLE", "warn_unstable", True, "1"),
    ],
)
def test_unset_config_env_vars(
    environment_variable: str, config_setting: str, value: Any, expected: str
) -> None:
    assert environment_variable in _POLARS_CFG_ENV_VARS

    with pl.Config(**{config_setting: value}):
        assert os.environ[environment_variable] == expected

    with pl.Config(**{config_setting: None}):  # type: ignore[arg-type]
        assert environment_variable not in os.environ
