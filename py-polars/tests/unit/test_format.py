from __future__ import annotations

from decimal import Decimal as D
from typing import TYPE_CHECKING, Any, Iterator

import pytest

import polars as pl
from polars import ComputeError

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType


@pytest.fixture(autouse=True)
def _environ() -> Iterator[None]:
    """Fixture to ensure we run with default Config settings during tests."""
    with pl.Config(restore_defaults=True):
        yield


@pytest.mark.parametrize(
    ("expected", "values"),
    [
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"Somelongstring…
]
""",
            ["Somelongstringto eeat wit me oundaf"],
            id="Long string",
        ),
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"😀😁😂😃😄😅😆😇😈😉😊😋😌😎…
]
""",
            ["😀😁😂😃😄😅😆😇😈😉😊😋😌😎😏😐😑😒😓"],
            id="Emojis",
        ),
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"yzäöüäöüäöüäö"
]
""",
            ["yzäöüäöüäöüäö"],
            id="Characters with accents",
        ),
        pytest.param(
            """shape: (100,)
Series: 'foo' [i64]
[
	0
	1
	2
	3
	4
	5
	6
	7
	8
	9
	10
	11
	…
	87
	88
	89
	90
	91
	92
	93
	94
	95
	96
	97
	98
	99
]
""",
            [*range(100)],
            id="Long series",
        ),
    ],
)
def test_fmt_series(
    capfd: pytest.CaptureFixture[str], expected: str, values: list[Any]
) -> None:
    s = pl.Series(name="foo", values=values)
    print(s)
    out, err = capfd.readouterr()
    assert out == expected


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        (
            [-127, -1, 0, 1, 127],
            pl.Int8,
            """shape: (5,)
Series: 'foo' [i8]
[
	-127
	-1
	0
	1
	127
]""",
        ),
        (
            [-32768, -1, 0, 1, 32767],
            pl.Int16,
            """shape: (5,)
Series: 'foo' [i16]
[
	-32,768
	-1
	0
	1
	32,767
]""",
        ),
        (
            [-2147483648, -1, 0, 1, 2147483647],
            pl.Int32,
            """shape: (5,)
Series: 'foo' [i32]
[
	-2,147,483,648
	-1
	0
	1
	2,147,483,647
]""",
        ),
        (
            [-9223372036854775808, -1, 0, 1, 9223372036854775807],
            pl.Int64,
            """shape: (5,)
Series: 'foo' [i64]
[
	-9,223,372,036,854,775,808
	-1
	0
	1
	9,223,372,036,854,775,807
]""",
        ),
    ],
)
def test_fmt_signed_int_thousands_sep(
    values: list[int], dtype: PolarsDataType, expected: str
) -> None:
    s = pl.Series(name="foo", values=values, dtype=dtype)
    with pl.Config(thousands_separator=True):
        assert str(s) == expected


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        (
            [0, 1, 127],
            pl.UInt8,
            """shape: (3,)
Series: 'foo' [u8]
[
	0
	1
	127
]""",
        ),
        (
            [0, 1, 32767],
            pl.UInt16,
            """shape: (3,)
Series: 'foo' [u16]
[
	0
	1
	32,767
]""",
        ),
        (
            [0, 1, 2147483647],
            pl.UInt32,
            """shape: (3,)
Series: 'foo' [u32]
[
	0
	1
	2,147,483,647
]""",
        ),
        (
            [0, 1, 9223372036854775807],
            pl.UInt64,
            """shape: (3,)
Series: 'foo' [u64]
[
	0
	1
	9,223,372,036,854,775,807
]""",
        ),
    ],
)
def test_fmt_unsigned_int_thousands_sep(
    values: list[int], dtype: PolarsDataType, expected: str
) -> None:
    s = pl.Series(name="foo", values=values, dtype=dtype)
    with pl.Config(thousands_separator=True):
        assert str(s) == expected


def test_fmt_float(capfd: pytest.CaptureFixture[str]) -> None:
    s = pl.Series(name="foo", values=[7.966e-05, 7.9e-05, 8.4666e-05, 8.00007966])
    print(s)
    out, err = capfd.readouterr()
    expected = """shape: (4,)
Series: 'foo' [f64]
[
	0.00008
	0.000079
	0.000085
	8.00008
]
"""
    assert out == expected


def test_duration_smallest_units() -> None:
    s = pl.Series(range(6), dtype=pl.Duration("us"))
    assert (
        str(s)
        == "shape: (6,)\nSeries: '' [duration[μs]]\n[\n\t0µs\n\t1µs\n\t2µs\n\t3µs\n\t4µs\n\t5µs\n]"
    )
    s = pl.Series(range(6), dtype=pl.Duration("ms"))
    assert (
        str(s)
        == "shape: (6,)\nSeries: '' [duration[ms]]\n[\n\t0ms\n\t1ms\n\t2ms\n\t3ms\n\t4ms\n\t5ms\n]"
    )
    s = pl.Series(range(6), dtype=pl.Duration("ns"))
    assert (
        str(s)
        == "shape: (6,)\nSeries: '' [duration[ns]]\n[\n\t0ns\n\t1ns\n\t2ns\n\t3ns\n\t4ns\n\t5ns\n]"
    )


def test_fmt_float_full() -> None:
    fmt_float_full = "shape: (1,)\nSeries: '' [f64]\n[\n\t1.230498095872587\n]"
    s = pl.Series([1.2304980958725870923])

    with pl.Config() as cfg:
        cfg.set_fmt_float("full")
        assert str(s) == fmt_float_full

    assert str(s) != fmt_float_full


def test_fmt_list_12188() -> None:
    # set max_items to 1 < 4(size of failed list) to touch the testing branch.
    with pl.Config(fmt_table_cell_list_len=1), pytest.raises(
        ComputeError, match="from `i64` to `u8` failed"
    ):
        pl.DataFrame(
            {
                "x": pl.int_range(250, 260, 1, eager=True),
            }
        ).with_columns(u8=pl.col("x").cast(pl.UInt8))


def test_date_list_fmt() -> None:
    df = pl.DataFrame(
        {
            "mydate": ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-05"],
            "index": [1, 2, 5, 5],
        }
    )

    df = df.with_columns(pl.col("mydate").str.strptime(pl.Date, "%Y-%m-%d"))
    assert (
        str(df.group_by("index", maintain_order=True).agg(pl.col("mydate"))["mydate"])
        == """shape: (3,)
Series: 'mydate' [list[date]]
[
	[2020-01-01]
	[2020-01-02]
	[2020-01-05, 2020-01-05]
]"""
    )


def test_fmt_series_cat_list() -> None:
    s = pl.Series(
        [
            ["a", "b"],
            ["b", "a"],
            ["b"],
        ],
    ).cast(pl.List(pl.Categorical))

    assert (
        str(s)
        == """shape: (3,)
Series: '' [list[cat]]
[
	["a", "b"]
	["b", "a"]
	["b"]
]"""
    )


def test_format_numeric_locale_options() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy"],
            "b": [100000.987654321, -234567.89],
            "c": [-11111111, 44444444444],
            "d": [D("12345.6789"), D("-9999999.99")],
        }
    )

    # note: numeric digit grouping looks much better
    # when right-aligned with fixed float precision
    with pl.Config(
        tbl_cell_numeric_alignment="RIGHT",
        thousands_separator=",",
        float_precision=3,
    ):
        print(df)
        assert (
            str(df)
            == """shape: (2, 4)
┌─────┬──────────────┬────────────────┬─────────────────┐
│ a   ┆            b ┆              c ┆               d │
│ --- ┆          --- ┆            --- ┆             --- │
│ str ┆          f64 ┆            i64 ┆    decimal[*,4] │
╞═════╪══════════════╪════════════════╪═════════════════╡
│ xx  ┆  100,000.988 ┆    -11,111,111 ┆     12,345.6789 │
│ yy  ┆ -234,567.890 ┆ 44,444,444,444 ┆ -9,999,999.9900 │
└─────┴──────────────┴────────────────┴─────────────────┘"""
        )

    # switch digit/decimal separators
    with pl.Config(
        decimal_separator=",",
        thousands_separator=".",
    ):
        assert (
            str(df)
            == """shape: (2, 4)
┌─────┬────────────────┬────────────────┬─────────────────┐
│ a   ┆ b              ┆ c              ┆ d               │
│ --- ┆ ---            ┆ ---            ┆ ---             │
│ str ┆ f64            ┆ i64            ┆ decimal[*,4]    │
╞═════╪════════════════╪════════════════╪═════════════════╡
│ xx  ┆ 100.000,987654 ┆ -11.111.111    ┆ 12.345,6789     │
│ yy  ┆ -234.567,89    ┆ 44.444.444.444 ┆ -9.999.999,9900 │
└─────┴────────────────┴────────────────┴─────────────────┘"""
        )

    # default (no digit grouping, standard digit/decimal separators)
    assert (
        str(df)
        == """shape: (2, 4)
┌─────┬───────────────┬─────────────┬───────────────┐
│ a   ┆ b             ┆ c           ┆ d             │
│ --- ┆ ---           ┆ ---         ┆ ---           │
│ str ┆ f64           ┆ i64         ┆ decimal[*,4]  │
╞═════╪═══════════════╪═════════════╪═══════════════╡
│ xx  ┆ 100000.987654 ┆ -11111111   ┆ 12345.6789    │
│ yy  ┆ -234567.89    ┆ 44444444444 ┆ -9999999.9900 │
└─────┴───────────────┴─────────────┴───────────────┘"""
    )
