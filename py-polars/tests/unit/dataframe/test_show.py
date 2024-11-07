import pytest

import polars as pl


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


@pl.Config(float_precision=8)
def test_df_show_float_precision(capsys: pytest.CaptureFixture[str]) -> None:
    from math import e, pi

    df = pl.DataFrame({"const": ["pi", "e"], "value": [pi, e]})

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


@pl.Config(fmt_str_lengths=20)
def test_df_show_fmt_str_lengths(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {
            "txt": [
                "Play it, Sam. Play 'As Time Goes By'.",
                "This is the beginning of a beautiful friendship.",
            ]
        }
    )

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


@pl.Config(fmt_table_cell_list_len=5)
def test_df_show_fmt_table_cell_list_len(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"nums": [list(range(10))]})

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


@pl.Config(tbl_cell_alignment="LEFT")
def test_df_show_tbl_cell_alignment(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame(
        {"column_abc": [1.0, 2.5, 5.0], "column_xyz": [True, False, True]}
    )

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


@pl.Config(tbl_cell_numeric_alignment="LEFT")
def test_df_show_tbl_cell_numeric_alignment(capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import date

    df = pl.DataFrame(
        {
            "abc": [11, 2, 333],
            "mno": [date(2023, 10, 29), None, date(2001, 7, 5)],
            "xyz": [True, False, None],
        }
    )

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


@pl.Config(tbl_formatting="UTF8_FULL")
def test_df_show_tbl_formatting(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

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


@pl.Config(tbl_cols=2)
def test_df_show_tbl_cols(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({str(i): [i] for i in range(10)})

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
