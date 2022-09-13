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
""",
            ["Somelongstringto eeat wit me oundaf"],
            id="Long string",
        ),
        pytest.param(
            """"shape: (1,)
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
Series: 'foo' [i32]
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
def test_fmt_series(capfd, expected, values):
    s = pl.Series(name="foo", values=values)
    print(s)
    out, err = capfd.readouterr()
    expected = """shape: (1,)
Series: 'foo' [str]
[
	"Somelongstring...
]
"""
    s = pl.Series


def test_fmt_float(capfd):
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
