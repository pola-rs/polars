from typing import Any, List

import pytest

import polars as pl


@pytest.mark.parametrize(
    ("expected", "values"),
    [
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"Somelongstring...
]
""",
            ["Somelongstringto eeat wit me oundaf"],
            id="Long string",
        ),
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"😀😁😂😃😄😅😆😇😈😉😊😋😌😎...
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
	...
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
    capfd: pytest.CaptureFixture[str], expected: str, values: List[Any]
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
