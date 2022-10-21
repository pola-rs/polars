from __future__ import annotations

import os
from typing import Iterator

import pytest

import polars as pl


@pytest.fixture(autouse=True)
def environ() -> Iterator[None]:
    """Fixture to restore the environment variables/state after the test."""
    with pl.StringCache(), pl.Config():
        yield


def test_ascii_tables() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    # note: expect to render ascii only within the given scope
    with pl.Config() as cfg:
        cfg.set_ascii_tables(True)
        assert (
            str(df) == "shape: (3, 3)\n"
            "+-----+-----+-----+\n"
            "| a   | b   | c   |\n"
            "| --- | --- | --- |\n"
            "| i64 | i64 | i64 |\n"
            "+=================+\n"
            "| 1   | 4   | 7   |\n"
            "|-----+-----+-----|\n"
            "| 2   | 5   | 8   |\n"
            "|-----+-----+-----|\n"
            "| 3   | 6   | 9   |\n"
            "+-----+-----+-----+"
        )

    # confirm back to utf8 default after scope-exit
    assert (
        str(df) == "shape: (3, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 4   ┆ 7   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 2   ┆ 5   ┆ 8   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 3   ┆ 6   ┆ 9   │\n"
        "└─────┴─────┴─────┘"
    )


def test_hide_header_elements() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    pl.Config.set_tbl_hide_column_data_types(True)
    assert (
        str(df) == "shape: (3, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 4   ┆ 7   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 2   ┆ 5   ┆ 8   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 3   ┆ 6   ┆ 9   │\n"
        "└─────┴─────┴─────┘"
    )

    pl.Config.set_tbl_hide_column_data_types(False).set_tbl_hide_column_names(True)
    assert (
        str(df) == "shape: (3, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 4   ┆ 7   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 2   ┆ 5   ┆ 8   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 3   ┆ 6   ┆ 9   │\n"
        "└─────┴─────┴─────┘"
    )


def test_set_tbl_cols() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    pl.Config.set_tbl_cols(1)
    assert str(df).split("\n")[2] == "│ a   ┆ ... │"
    pl.Config.set_tbl_cols(2)
    assert str(df).split("\n")[2] == "│ a   ┆ ... ┆ c   │"
    pl.Config.set_tbl_cols(3)
    assert str(df).split("\n")[2] == "│ a   ┆ b   ┆ c   │"

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
    )
    pl.Config.set_tbl_cols(2)
    assert str(df).split("\n")[2] == "│ a   ┆ ... ┆ d   │"
    pl.Config.set_tbl_cols(3)
    assert str(df).split("\n")[2] == "│ a   ┆ b   ┆ ... ┆ d   │"
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
        "│ ... ┆ ... ┆ ... │\n"
        "└─────┴─────┴─────┘"
    )
    assert str(ser) == "shape: (5,)\n" "Series: 'ser' [i64]\n" "[\n" "\t...\n" "]"

    pl.Config.set_tbl_rows(1)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ ... ┆ ... ┆ ... │\n"
        "└─────┴─────┴─────┘"
    )
    assert (
        str(ser) == "shape: (5,)\n" "Series: 'ser' [i64]\n" "[\n" "\t1\n" "\t...\n" "]"
    )

    pl.Config.set_tbl_rows(2)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ ... ┆ ... ┆ ... │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 4   ┆ 8   ┆ 12  │\n"
        "└─────┴─────┴─────┘"
    )
    assert (
        str(ser) == "shape: (5,)\n"
        "Series: 'ser' [i64]\n"
        "[\n"
        "\t1\n"
        "\t...\n"
        "\t5\n"
        "]"
    )

    pl.Config.set_tbl_rows(3)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ ... ┆ ... ┆ ... │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 3   ┆ 7   ┆ 11  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 4   ┆ 8   ┆ 12  │\n"
        "└─────┴─────┴─────┘"
    )
    assert (
        str(ser) == "shape: (5,)\n"
        "Series: 'ser' [i64]\n"
        "[\n"
        "\t1\n"
        "\t...\n"
        "\t4\n"
        "\t5\n"
        "]"
    )

    pl.Config.set_tbl_rows(4)
    assert (
        str(df) == "shape: (4, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ --- ┆ --- ┆ --- │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 5   ┆ 9   │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 2   ┆ 6   ┆ 10  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 3   ┆ 7   ┆ 11  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 4   ┆ 8   ┆ 12  │\n"
        "└─────┴─────┴─────┘"
    )
    assert (
        str(ser) == "shape: (5,)\n"
        "Series: 'ser' [i64]\n"
        "[\n"
        "\t1\n"
        "\t2\n"
        "\t...\n"
        "\t4\n"
        "\t5\n"
        "]"
    )

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
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ ... ┆ ... ┆ ... │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 4   ┆ 9   ┆ 14  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 5   ┆ 10  ┆ 15  │\n"
        "└─────┴─────┴─────┘"
    )

    pl.Config.set_tbl_rows(-1)
    assert (
        str(ser) == "shape: (5,)\n"
        "Series: 'ser' [i64]\n"
        "[\n"
        "\t1\n"
        "\t2\n"
        "\t3\n"
        "\t4\n"
        "\t5\n"
        "]"
    )

    pl.Config.set_tbl_hide_dtype_separator(True)
    assert (
        str(df) == "shape: (5, 3)\n"
        "┌─────┬─────┬─────┐\n"
        "│ a   ┆ b   ┆ c   │\n"
        "│ i64 ┆ i64 ┆ i64 │\n"
        "╞═════╪═════╪═════╡\n"
        "│ 1   ┆ 6   ┆ 11  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 2   ┆ 7   ┆ 12  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 3   ┆ 8   ┆ 13  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
        "│ 4   ┆ 9   ┆ 14  │\n"
        "├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤\n"
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
    with pl.Config() as cfg:
        cfg.set_tbl_hide_dtype_separator(True)
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
    with pl.Config() as cfg:
        cfg.set_tbl_formatting("NOTHING").set_tbl_hide_column_data_types(True)
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


def test_set_tbl_width_chars() -> None:
    df = pl.DataFrame(
        {
            "a really long col": [1, 2, 3],
            "b": ["", "this is a string value that will be truncated", None],
            "this is 10": [4, 5, 6],
        }
    )
    assert max(len(line) for line in str(df).split("\n")) == 72

    pl.Config.set_tbl_width_chars(60)
    assert max(len(line) for line in str(df).split("\n")) == 60

    # formula for determining min width is
    # sum(max(min(header.len, 12), 5)) + header.len + 1
    # so we end up with 12+5+10+4 = 31

    pl.Config.set_tbl_width_chars(0)
    assert max(len(line) for line in str(df).split("\n")) == 31


def test_shape_below_table_and_inlined_dtype() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    pl.Config.set_tbl_column_data_type_inline(True).set_tbl_dataframe_shape_below(True)
    pl.Config.set_tbl_formatting("UTF8_FULL_CONDENSED")
    assert (
        str(df) == ""
        "┌─────────┬─────────┬─────────┐\n"
        "│ a (i64) ┆ b (i64) ┆ c (i64) │\n"
        "╞═════════╪═════════╪═════════╡\n"
        "│ 1       ┆ 3       ┆ 5       │\n"
        "│ 2       ┆ 4       ┆ 6       │\n"
        "└─────────┴─────────┴─────────┘\n"
        "shape: (2, 3)"
    )

    pl.Config.set_tbl_dataframe_shape_below(False)
    assert (
        str(df) == "shape: (2, 3)\n"
        "┌─────────┬─────────┬─────────┐\n"
        "│ a (i64) ┆ b (i64) ┆ c (i64) │\n"
        "╞═════════╪═════════╪═════════╡\n"
        "│ 1       ┆ 3       ┆ 5       │\n"
        "│ 2       ┆ 4       ┆ 6       │\n"
        "└─────────┴─────────┴─────────┘"
    )

    pl.Config.set_tbl_column_data_type_inline(False).set_tbl_cell_alignment("RIGHT")
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


def test_string_cache() -> None:
    df1 = pl.DataFrame({"a": ["foo", "bar", "ham"], "b": [1, 2, 3]})
    df2 = pl.DataFrame({"a": ["foo", "spam", "eggs"], "c": [3, 2, 2]})

    # ensure cache is off when casting to categorical; the join will fail
    pl.toggle_string_cache(False)
    assert pl.using_string_cache() is False

    df1a = df1.with_column(pl.col("a").cast(pl.Categorical))
    df2a = df2.with_column(pl.col("a").cast(pl.Categorical))
    with pytest.raises(pl.ComputeError):
        _ = df1a.join(df2a, on="a", how="inner")

    # now turn on the cache
    pl.toggle_string_cache(True)
    assert pl.using_string_cache() is True

    df1b = df1.with_column(pl.col("a").cast(pl.Categorical))
    df2b = df2.with_column(pl.col("a").cast(pl.Categorical))
    out = df1b.join(df2b, on="a", how="inner")
    assert out.frame_equal(pl.DataFrame({"a": ["foo"], "b": [1], "c": [3]}))


def test_config_load_save() -> None:
    # set some config options
    pl.Config.with_columns_kwargs = True
    pl.Config.set_verbose(True)
    assert os.environ["POLARS_VERBOSE"] == "1"

    cfg = pl.Config.save()
    assert isinstance(cfg, str)
    assert "POLARS_VERBOSE" in pl.Config.state(if_set=True)

    # unset the saved options
    pl.Config.with_columns_kwargs = False
    pl.Config.set_verbose(False)
    assert os.environ["POLARS_VERBOSE"] == "0"

    # now load back from config...
    pl.Config.load(cfg)

    # ...and confirm the saved options were set
    assert os.environ["POLARS_VERBOSE"] == "1"
    assert pl.Config.with_columns_kwargs is True

    # restore explicitly-set config options (unsets from env)
    pl.Config.restore_defaults()
    assert "POLARS_VERBOSE" not in pl.Config.state(if_set=True)
    assert "POLARS_VERBOSE" in pl.Config.state()
    assert os.environ.get("POLARS_VERBOSE") is None
    assert pl.Config.with_columns_kwargs is False


def test_config_scope() -> None:
    pl.Config.set_verbose(False)
    pl.Config.set_tbl_cols(8)

    initial_state = pl.Config.state()

    with pl.Config() as cfg:
        cfg.set_verbose(True).set_tbl_hide_dtype_separator(True).set_ascii_tables()

        new_state_entries = set(
            {
                "POLARS_FMT_MAX_COLS": "8",
                "POLARS_FMT_TABLE_FORMATTING": "ASCII_FULL",
                "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR": "1",
                "POLARS_VERBOSE": "1",
            }.items()
        )
        assert set(initial_state.items()) != new_state_entries
        assert new_state_entries.issubset(set(cfg.state().items()))

    # expect scope-exit to restore original state
    assert pl.Config.state() == initial_state
