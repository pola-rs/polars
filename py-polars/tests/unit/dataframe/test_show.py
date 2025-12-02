import pytest

import polars as pl
from polars.config import TableFormatNames


def test_df_show_default(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5, 6, 7],
            "bar": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )

    df.show()
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (5, 2)
┌─────┬─────┐
│ foo ┆ bar │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ b   │
│ 3   ┆ c   │
│ 4   ┆ d   │
│ 5   ┆ e   │
└─────┴─────┘
"""
    )


def test_df_show_positive_limit(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5, 6, 7],
            "bar": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )

    df.show(3)
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (3, 2)
┌─────┬─────┐
│ foo ┆ bar │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ b   │
│ 3   ┆ c   │
└─────┴─────┘
"""
    )


def test_df_show_negative_limit(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5, 6, 7],
            "bar": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )

    df.show(-5)
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (2, 2)
┌─────┬─────┐
│ foo ┆ bar │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ b   │
└─────┴─────┘
"""
    )


def test_df_show_no_limit(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5, 6, 7],
            "bar": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )

    df.show(limit=None)
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (7, 2)
┌─────┬─────┐
│ foo ┆ bar │
│ --- ┆ --- │
│ i64 ┆ str │
╞═════╪═════╡
│ 1   ┆ a   │
│ 2   ┆ b   │
│ 3   ┆ c   │
│ 4   ┆ d   │
│ 5   ┆ e   │
│ 6   ┆ f   │
│ 7   ┆ g   │
└─────┴─────┘
"""
    )


def test_df_show_ascii_tables(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(ascii_tables=False):
        df.show(ascii_tables=True)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 2)
+-----+-------+
| abc | xyz   |
| --- | ---   |
| f64 | bool  |
+=============+
| 1.0 | true  |
| 2.5 | false |
| 5.0 | true  |
+-----+-------+
"""
        )


@pytest.mark.parametrize(
    ("ascii_tables", "tbl_formatting"),
    [
        (True, "ASCII_FULL_CONDENSED"),
        (True, "UTF8_FULL_CONDENSED"),
        (False, "ASCII_FULL_CONDENSED"),
        (False, "UTF8_FULL_CONDENSED"),
    ],
)
def test_df_show_cannot_set_ascii_tables_and_tbl_formatting(
    ascii_tables: bool, tbl_formatting: TableFormatNames
) -> None:
    df = pl.DataFrame()

    with pytest.raises(ValueError):
        df.show(ascii_tables=ascii_tables, tbl_formatting=tbl_formatting)


def test_df_show_decimal_separator(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"v": [9876.54321, 1010101.0, -123456.78]})

    with pl.Config(decimal_separator="."):
        df.show(decimal_separator=",")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 1)
┌────────────┐
│ v          │
│ ---        │
│ f64        │
╞════════════╡
│ 9876,54321 │
│ 1,010101e6 │
│ -123456,78 │
└────────────┘
"""
        )


def test_df_show_thousands_separator(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"v": [9876.54321, 1010101.0, -123456.78]})

    with pl.Config(thousands_separator="."):
        df.show(thousands_separator=" ")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 1)
┌─────────────┐
│ v           │
│ ---         │
│ f64         │
╞═════════════╡
│ 9 876.54321 │
│ 1.010101e6  │
│ -123 456.78 │
└─────────────┘
"""
        )


def test_df_show_float_precision(capsys: pytest.CaptureFixture[str]) -> None:
    from math import e, pi

    df = pl.DataFrame({"const": ["pi", "e"], "value": [pi, e]})

    with pl.Config(float_precision=8):
        df.show(float_precision=15)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (2, 2)
┌───────┬───────────────────┐
│ const ┆ value             │
│ ---   ┆ ---               │
│ str   ┆ f64               │
╞═══════╪═══════════════════╡
│ pi    ┆ 3.141592653589793 │
│ e     ┆ 2.718281828459045 │
└───────┴───────────────────┘
"""
        )

        df.show(float_precision=3)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (2, 2)
┌───────┬───────┐
│ const ┆ value │
│ ---   ┆ ---   │
│ str   ┆ f64   │
╞═══════╪═══════╡
│ pi    ┆ 3.142 │
│ e     ┆ 2.718 │
└───────┴───────┘
"""
        )


def test_df_show_fmt_float(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"num": [1.2304980958725870923, 1e6, 1e-8]})

    with pl.Config(fmt_float="full"):
        df.show(fmt_float="mixed")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 1)
┌───────────┐
│ num       │
│ ---       │
│ f64       │
╞═══════════╡
│ 1.230498  │
│ 1e6       │
│ 1.0000e-8 │
└───────────┘
"""
        )


def test_df_show_fmt_str_lengths(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "txt": [
                "Play it, Sam. Play 'As Time Goes By'.",
                "This is the beginning of a beautiful friendship.",
            ]
        }
    )

    with pl.Config(fmt_str_lengths=20):
        df.show(fmt_str_lengths=10)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (2, 1)
┌─────────────┐
│ txt         │
│ ---         │
│ str         │
╞═════════════╡
│ Play it, S… │
│ This is th… │
└─────────────┘
"""
        )

        df.show(fmt_str_lengths=50)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (2, 1)
┌──────────────────────────────────────────────────┐
│ txt                                              │
│ ---                                              │
│ str                                              │
╞══════════════════════════════════════════════════╡
│ Play it, Sam. Play 'As Time Goes By'.            │
│ This is the beginning of a beautiful friendship. │
└──────────────────────────────────────────────────┘
"""
        )


def test_df_show_fmt_table_cell_list_len(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"nums": [list(range(10))]})

    with pl.Config(fmt_table_cell_list_len=5):
        df.show(fmt_table_cell_list_len=2)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (1, 1)
┌───────────┐
│ nums      │
│ ---       │
│ list[i64] │
╞═══════════╡
│ [0, … 9]  │
└───────────┘
"""
        )

        df.show(fmt_table_cell_list_len=8)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (1, 1)
┌────────────────────────────┐
│ nums                       │
│ ---                        │
│ list[i64]                  │
╞════════════════════════════╡
│ [0, 1, 2, 3, 4, 5, 6, … 9] │
└────────────────────────────┘
"""
        )


def test_df_show_tbl_cell_alignment(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {"column_abc": [1.0, 2.5, 5.0], "column_xyz": [True, False, True]}
    )

    with pl.Config(tbl_cell_alignment="LEFT"):
        df.show(tbl_cell_alignment="RIGHT")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 2)
┌────────────┬────────────┐
│ column_abc ┆ column_xyz │
│        --- ┆        --- │
│        f64 ┆       bool │
╞════════════╪════════════╡
│        1.0 ┆       true │
│        2.5 ┆      false │
│        5.0 ┆       true │
└────────────┴────────────┘
"""
        )


def test_df_show_tbl_cell_numeric_alignment(capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import date

    df = pl.DataFrame(
        {
            "abc": [11, 2, 333],
            "mno": [date(2023, 10, 29), None, date(2001, 7, 5)],
            "xyz": [True, False, None],
        }
    )

    with pl.Config(tbl_cell_numeric_alignment="LEFT"):
        df.show(tbl_cell_numeric_alignment="RIGHT")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 3)
┌─────┬────────────┬───────┐
│ abc ┆ mno        ┆ xyz   │
│ --- ┆ ---        ┆ ---   │
│ i64 ┆ date       ┆ bool  │
╞═════╪════════════╪═══════╡
│  11 ┆ 2023-10-29 ┆ true  │
│   2 ┆ null       ┆ false │
│ 333 ┆ 2001-07-05 ┆ null  │
└─────┴────────────┴───────┘
"""
        )


def test_df_show_tbl_cols(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({str(i): [i] for i in range(10)})

    with pl.Config(tbl_cols=2):
        df.show(tbl_cols=3)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (1, 10)
┌─────┬─────┬───┬─────┐
│ 0   ┆ 1   ┆ … ┆ 9   │
│ --- ┆ --- ┆   ┆ --- │
│ i64 ┆ i64 ┆   ┆ i64 │
╞═════╪═════╪═══╪═════╡
│ 0   ┆ 1   ┆ … ┆ 9   │
└─────┴─────┴───┴─────┘
"""
        )

        df.show(tbl_cols=7)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (1, 10)
┌─────┬─────┬─────┬─────┬───┬─────┬─────┬─────┐
│ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 7   ┆ 8   ┆ 9   │
│ --- ┆ --- ┆ --- ┆ --- ┆   ┆ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 ┆ i64 ┆   ┆ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╪═════╪═══╪═════╪═════╪═════╡
│ 0   ┆ 1   ┆ 2   ┆ 3   ┆ … ┆ 7   ┆ 8   ┆ 9   │
└─────┴─────┴─────┴─────┴───┴─────┴─────┴─────┘
"""
        )


def test_df_show_tbl_column_data_type_inline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(tbl_column_data_type_inline=False):
        df.show(tbl_column_data_type_inline=True)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 2)
┌───────────┬────────────┐
│ abc (f64) ┆ xyz (bool) │
╞═══════════╪════════════╡
│ 1.0       ┆ true       │
│ 2.5       ┆ false      │
│ 5.0       ┆ true       │
└───────────┴────────────┘
"""
        )


def test_df_show_tbl_dataframe_shape_below(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(tbl_dataframe_shape_below=False):
        df.show(tbl_dataframe_shape_below=True)
        out, _ = capsys.readouterr()
        assert out == (
            "┌─────┬───────┐\n"
            "│ abc ┆ xyz   │\n"
            "│ --- ┆ ---   │\n"
            "│ f64 ┆ bool  │\n"
            "╞═════╪═══════╡\n"
            "│ 1.0 ┆ true  │\n"
            "│ 2.5 ┆ false │\n"
            "│ 5.0 ┆ true  │\n"
            "└─────┴───────┘\n"
            "shape: (3, 2)\n"
        )


def test_df_show_tbl_formatting(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    with pl.Config(tbl_formatting="UTF8_FULL"):
        df.show(tbl_formatting="ASCII_FULL")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 3)
+-----+-----+-----+
| a   | b   | c   |
| --- | --- | --- |
| i64 | i64 | i64 |
+=================+
| 1   | 4   | 7   |
|-----+-----+-----|
| 2   | 5   | 8   |
|-----+-----+-----|
| 3   | 6   | 9   |
+-----+-----+-----+
"""
        )

        df.show(tbl_formatting="MARKDOWN")
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 3)
| a   | b   | c   |
| --- | --- | --- |
| i64 | i64 | i64 |
|-----|-----|-----|
| 1   | 4   | 7   |
| 2   | 5   | 8   |
| 3   | 6   | 9   |
"""
        )


def test_df_show_tbl_hide_column_data_types(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(tbl_hide_column_data_types=False):
        df.show(tbl_hide_column_data_types=True)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 2)
┌─────┬───────┐
│ abc ┆ xyz   │
╞═════╪═══════╡
│ 1.0 ┆ true  │
│ 2.5 ┆ false │
│ 5.0 ┆ true  │
└─────┴───────┘
"""
        )


def test_df_show_tbl_hide_column_names(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(tbl_hide_column_names=False):
        df.show(tbl_hide_column_names=True)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 2)
┌─────┬───────┐
│ f64 ┆ bool  │
╞═════╪═══════╡
│ 1.0 ┆ true  │
│ 2.5 ┆ false │
│ 5.0 ┆ true  │
└─────┴───────┘
"""
        )


def test_df_show_tbl_hide_dtype_separator(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(tbl_hide_dtype_separator=False):
        df.show(tbl_hide_dtype_separator=True)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (3, 2)
┌─────┬───────┐
│ abc ┆ xyz   │
│ f64 ┆ bool  │
╞═════╪═══════╡
│ 1.0 ┆ true  │
│ 2.5 ┆ false │
│ 5.0 ┆ true  │
└─────┴───────┘
"""
        )


def test_df_show_tbl_hide_dataframe_shape(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"abc": [1.0, 2.5, 5.0], "xyz": [True, False, True]})

    with pl.Config(tbl_hide_dataframe_shape=False):
        df.show(tbl_hide_dataframe_shape=True)
        out, _ = capsys.readouterr()
        assert out == (
            "┌─────┬───────┐\n"
            "│ abc ┆ xyz   │\n"
            "│ --- ┆ ---   │\n"
            "│ f64 ┆ bool  │\n"
            "╞═════╪═══════╡\n"
            "│ 1.0 ┆ true  │\n"
            "│ 2.5 ┆ false │\n"
            "│ 5.0 ┆ true  │\n"
            "└─────┴───────┘\n"
        )


def test_df_show_tbl_width_chars(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "id": ["SEQ1", "SEQ2"],
            "seq": ["ATGATAAAGGAG", "GCAACGCATATA"],
        }
    )

    with pl.Config(tbl_width_chars=100):
        df.show(tbl_width_chars=12)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (2, 2)
┌─────┬─────┐
│ id  ┆ seq │
│ --- ┆ --- │
│ str ┆ str │
╞═════╪═════╡
│ SEQ ┆ ATG │
│ 1   ┆ ATA │
│     ┆ AAG │
│     ┆ GAG │
│ SEQ ┆ GCA │
│ 2   ┆ ACG │
│     ┆ CAT │
│     ┆ ATA │
└─────┴─────┘
"""
        )


def test_df_show_trim_decimal_zeros(capsys: pytest.CaptureFixture[str]) -> None:
    from decimal import Decimal as D

    df = pl.DataFrame(
        data={"d": [D("1.01000"), D("-5.67890")]},
        schema={"d": pl.Decimal(scale=5)},
    )

    with pl.Config(trim_decimal_zeros=False):
        df.show(trim_decimal_zeros=True)
        out, _ = capsys.readouterr()
        assert (
            out
            == """shape: (2, 1)
┌───────────────┐
│ d             │
│ ---           │
│ decimal[38,5] │
╞═══════════════╡
│ 1.01          │
│ -5.6789       │
└───────────────┘
"""
        )
