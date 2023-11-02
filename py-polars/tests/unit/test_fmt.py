from __future__ import annotations

from typing import Any, Iterator

import pytest

import polars as pl
from polars import ComputeError


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
