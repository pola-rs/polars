from decimal import Decimal

import pytest

import polars as pl


def test_series_show_default(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5, 6, 7])

    s.show()
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (5,)
Series: 'a' [i64]
[
	1
	2
	3
	4
	5
]
"""
    )


def test_series_show_positive_limit(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5, 6, 7])

    s.show(3)
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (3,)
Series: 'a' [i64]
[
	1
	2
	3
]
"""
    )


def test_series_show_negative_limit(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5, 6, 7])

    s.show(-5)
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (2,)
Series: 'a' [i64]
[
	1
	2
]
"""
    )


def test_series_show_negative_limit_larger_than_len(
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = pl.Series("a", [1, 2, 3])

    s.show(-5)
    out, _ = capsys.readouterr()
    assert (
        out
        == """shape: (0,)
Series: 'a' [i64]
[
]
"""
    )


def test_series_show_no_limit(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", range(30))

    s.show(limit=None)
    out, _ = capsys.readouterr()
    assert out.startswith("shape: (30,)\n")
    assert "…" not in out
    assert out.count("\n") == 30 + 4


def test_series_show_limit_larger_than_default_tbl_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = pl.Series("a", range(30))

    s.show(25)
    out, _ = capsys.readouterr()
    assert out.startswith("shape: (25,)\n")
    assert "…" not in out


def test_series_show_decimal_separator(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1.5])

    s.show(decimal_separator=",")
    out, _ = capsys.readouterr()
    assert "\t1,5\n" in out


def test_series_show_thousands_separator(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1234567])

    s.show(thousands_separator=True)
    out, _ = capsys.readouterr()
    assert "\t1,234,567\n" in out


def test_series_show_float_precision(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1.23456789])

    s.show(float_precision=2)
    out, _ = capsys.readouterr()
    assert "\t1.23\n" in out


def test_series_show_fmt_float(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1e-12])

    s.show(fmt_float="full")
    out, _ = capsys.readouterr()
    assert "\t0.000000000001\n" in out


def test_series_show_fmt_str_lengths(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", ["abcdefghijklmnopqrstuvwxyz"])

    s.show(fmt_str_lengths=5)
    out, _ = capsys.readouterr()
    assert '\t"abcde…\n' in out


def test_series_show_fmt_table_cell_list_len(
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = pl.Series("a", [[1, 2, 3, 4, 5, 6]])

    s.show(fmt_table_cell_list_len=2)
    out, _ = capsys.readouterr()
    assert "\t[1, … 6]\n" in out


def test_series_show_trim_decimal_zeros(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [Decimal("1.50")], dtype=pl.Decimal(10, 2))

    s.show()
    out, _ = capsys.readouterr()
    assert "\t1.5\n" in out

    s.show(trim_decimal_zeros=False)
    out, _ = capsys.readouterr()
    assert "\t1.50\n" in out


def test_series_show_does_not_leak_config(capsys: pytest.CaptureFixture[str]) -> None:
    s = pl.Series("a", [1.23456789])

    s.show(float_precision=2)
    capsys.readouterr()

    print(s)
    out, _ = capsys.readouterr()
    assert "\t1.234568\n" in out
