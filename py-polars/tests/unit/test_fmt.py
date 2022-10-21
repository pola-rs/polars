from typing import Any, List

import pytest

import polars as pl


@pytest.mark.parametrize(
    "expected, values",
    [
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"Somelongstring...
]
""",  # noqa: E101, W191
            ["Somelongstringto eeat wit me oundaf"],
            id="Long string",
        ),
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"😀😁😂😃😄😅😆😇😈😉😊😋😌😎...
]
""",  # noqa: E101, W191
            ["😀😁😂😃😄😅😆😇😈😉😊😋😌😎😏😐😑😒😓"],
            id="Emojis",
        ),
        pytest.param(
            """shape: (1,)
Series: 'foo' [str]
[
	"yzäöüäöüäöüäö"
]
""",  # noqa: E101, W191
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
""",  # noqa: E101, W191
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
"""  # noqa: E101, W191
    assert out == expected
