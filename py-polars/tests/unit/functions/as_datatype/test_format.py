from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal


def test_format() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt")])
    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]


def test_format_empty_chunks_25525() -> None:
    s = pl.concat(
        [
            pl.Series("a", ["p"]),
            pl.Series("a", ["x", "y"]),
            pl.Series("a", ["q"]),
            pl.Series("a", ["z", None]),
        ],
        rechunk=False,
    )
    df = pl.DataFrame([s, pl.Series("k", [0, 1, 1, 0, 1, 1])])
    df = df.filter(df["k"] == 1).with_columns(b=pl.Series(["1", "2", "3", "4"]))

    out = df.select(pl.format("<{}>", pl.col("a")))
    assert out.to_series().to_list() == ["<x>", "<y>", "<z>", None]

    out = df.select(pl.format("<{}-{}-{}>", pl.col("b"), pl.lit("s"), pl.col("a")))
    assert out.to_series().to_list() == ["<1-s-x>", "<2-s-y>", "<3-s-z>", None]


INTS = [0, 7, -42, 1234, -1234567, 2**63 - 1, -(2**63)]
FLOATS = [0.0, -0.0, 2.5, 0.125, 2.675, -1234.5, 1e-5, 123456.789, 1e22, float("inf")]


@pytest.mark.parametrize(
    "spec",
    [
        "d",
        "5",
        "<6",
        "*^9",
        "+",
        " d",
        "08",
        "=+8",
        ",",
        "012,",
        "_",
        "x",
        "#X",
        "#010b",
        "_o",
        "#_x",
        ".2f",
        "e",
    ],
)
def test_format_spec_int(spec: str) -> None:
    out = pl.select(pl.format(f"{{:{spec}}}", pl.Series(INTS)))
    assert out.to_series().to_list() == [format(v, spec) for v in INTS]


@pytest.mark.parametrize(
    "spec",
    [
        ".2f",
        "f",
        ".0f",
        "#.0f",
        "+.1f",
        "010.3f",
        ",.2f",
        "e",
        ".3E",
        "#.0e",
        "g",
        ".3g",
        "#g",
        "G",
        ".1%",
        "%",
        ".3",
        "^+14.2e",
        "F",
    ],
)
def test_format_spec_float(spec: str) -> None:
    out = pl.select(pl.format(f"{{:{spec}}}", pl.Series(FLOATS)))
    assert out.to_series().to_list() == [format(v, spec) for v in FLOATS]


def test_format_spec_str() -> None:
    values = ["a", "héllo", "", None]
    out = pl.select(
        a=pl.format("[{:>6}]", pl.Series(values)),
        b=pl.format("[{:*^7.3}]", pl.Series(values)),
        c=pl.format("[{:05s}]", pl.Series(values)),
    )
    assert out.rows() == [
        ("[     a]", "[***a***]", "[a0000]"),
        ("[ héllo]", "[**hél**]", "[héllo]"),
        ("[      ]", "[*******]", "[00000]"),
        (None, None, None),
    ]


def test_format_spec_other_dtypes() -> None:
    df = pl.DataFrame(
        {
            "f": pl.Series([1.5, -2.0, None], dtype=pl.Float32),
            "d": [date(2024, 1, 2), None, date(1999, 12, 31)],
            "b": [True, False, None],
            "dec": pl.Series(
                [Decimal("2.5"), Decimal("-0.125"), Decimal("1234.005")],
                dtype=pl.Decimal(10, 3),
            ),
            "u": pl.Series([255, 0, 2**64 - 1], dtype=pl.UInt64),
        }
    )
    out = df.select(
        f=pl.format("{:>6}|{:+,}", "f", "f"),
        d=pl.format("{:>12}", "d"),
        b=pl.format("{:<6}|{:d}", "b", "b"),
        dec=pl.format("{:.2f}|{:,}|{:.0%}|{:.1e}", "dec", "dec", "dec", "dec"),
        u=pl.format("{:#x}", "u"),
    )
    assert out.rows() == [
        ("   1.5|+1.5", "  2024-01-02", "true  |1", "2.50|2.500|250%|2.5e+00", "0xff"),
        ("  -2.0|-2.0", None, "false |0", "-0.12|-0.125|-12%|-1.2e-01", "0x0"),
        (
            None,
            "  1999-12-31",
            None,
            "1234.00|1,234.005|123400%|1.2e+03",
            "0xffffffffffffffff",
        ),
    ]


def test_format_spec_placeholders() -> None:
    df = pl.DataFrame({"a": [1.25, 10.0], "b": ["x", "yz"]})
    out = df.select(
        pl.format("{1:>3}={0:.1f} {{{0:+}}}", "a", "b").alias("indexed"),
        pl.format("{b:>3}={a:.1f}").alias("named"),
        pl.format("{:}|{:}", "a", "b").alias("empty_spec"),
    )
    assert out.rows() == [
        ("  x=1.2 {+1.25}", "  x=1.2", "1.25|x"),
        (" yz=10.0 {+10.0}", " yz=10.0", "10.0|yz"),
    ]


def test_format_spec_engines() -> None:
    lf = pl.LazyFrame({"g": [1, 2, 1, 2], "x": [1.0, 2.5, -3.25, None]})
    q = lf.select(
        a=pl.format("{:>8.2f}", "x"),
        b=pl.format("{:+}", pl.col("x").sum().over("g")),
    )
    expected = pl.DataFrame(
        {
            "a": ["    1.00", "    2.50", "   -3.25", None],
            "b": ["-2.25", "+2.5", "-2.25", "+2.5"],
        }
    )
    assert_frame_equal(q.collect(), expected)
    assert_frame_equal(q.collect(engine="streaming"), expected)


@pytest.mark.parametrize(
    ("spec", "dtype", "match"),
    [
        (".2d", pl.Int64, "a precision is not allowed"),
        ("d", pl.Float64, "format code 'd' is not supported for f64"),
        ("s", pl.Int32, "format code 's' is not supported for i32"),
        ("+", pl.String, "a sign is not allowed"),
        ("=5", pl.Date, "'=' alignment is not allowed"),
        (",x", pl.Int64, "',' is not allowed with format code 'x'"),
        ("+c", pl.Int64, "format code 'c' does not allow a sign"),
        ("f", pl.String, "format code 'f' is not supported for str"),
    ],
)
def test_format_spec_dtype_errors(spec: str, dtype: pl.DataType, match: str) -> None:
    lf = pl.LazyFrame({"a": []}, schema={"a": dtype})
    with pytest.raises(InvalidOperationError, match=match):
        lf.select(pl.format(f"{{:{spec}}}", "a")).collect_schema()


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("y", "invalid format specifier 'y'"),
        (".f", "missing precision in format specifier '.f'"),
        ("5.2fx", "invalid format specifier '5.2fx'"),
        ("99999999999999999999999", "number too large"),
        (".2147483648g", "number too large"),
    ],
)
def test_format_spec_parse_errors(spec: str, match: str) -> None:
    with pytest.raises(InvalidOperationError, match=match):
        pl.format(f"{{:{spec}}}", pl.col("a"))


def test_format_spec_edge_cases() -> None:
    zero = pl.Series([Decimal("0"), Decimal("-0.5")], dtype=pl.Decimal(3, 1))
    out = pl.select(
        a=pl.format("{:.2%}", zero),
        b=pl.format("{:#.0%}", pl.lit(1e308)),
        c=pl.format("{:.70000f}|{:.70000e}", pl.lit(0.5), pl.lit(0.5)).str.len_chars(),
        d=pl.format("{:.70000g}|{:.70000}", pl.lit(1e-4), pl.lit(1e-300)),
        e=pl.format("{:#}|{:#}|{:#>8}", pl.lit(1e20), pl.lit(1.5), zero),
    )
    long = format(1e-4, ".70000g") + "|" + format(1e-300, ".70000")
    assert out.rows() == [
        ("0.00%", "inf%", 140009, long, "1.e+20|1.5|#####0.0"),
        ("-50.00%", "inf%", 140009, long, "1.e+20|1.5|####-0.5"),
    ]
    whole = pl.Series([Decimal("3"), Decimal("-12")], dtype=pl.Decimal(4, 0))
    assert pl.select(pl.format("{:#}", whole)).to_series().to_list() == ["3.", "-12."]


def test_format_spec_invalid_code_point() -> None:
    with pytest.raises(ComputeError, match="invalid Unicode code point: -1"):
        pl.select(pl.format("{:c}", pl.Series([65, -1])))
    assert pl.select(pl.format("{:c}{:c}", pl.lit(72), pl.lit(105))).item() == "Hi"
    # a row with a null is null, so its invalid code point is not formatted
    df = pl.DataFrame({"s": ["x", None], "code": [65, -1]})
    res = df.select(pl.format("{}{:c}", "s", "code")).to_series()
    assert res.to_list() == ["xA", None]
