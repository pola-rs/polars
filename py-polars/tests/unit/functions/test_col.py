from __future__ import annotations

import polars as pl
from polars import col
from polars.datatypes.group import NUMERIC_DTYPES
from polars.testing import assert_frame_equal


def test_col_as_attribute() -> None:
    df = pl.DataFrame({"lower": 1, "UPPER": 2, "_underscored": 3})

    result = df.select(col.lower, col.UPPER, col._underscored)
    expected = df.select("lower", "UPPER", "_underscored")
    assert_frame_equal(result, expected)


def test_col_as_attribute_edge_cases() -> None:
    df = pl.DataFrame(
        {
            "__misc": "x",
            "__wrapped__col": "y",
            "_other__col__": "z",
        }
    )
    for select_cols in (
        (pl.col("_other__col__"), pl.col("__wrapped__col"), pl.col("__misc")),
        (pl.col._other__col__, pl.col.__wrapped__col, pl.col.__misc),
    ):
        assert df.select(select_cols).columns == [
            "_other__col__",
            "__wrapped__col",
            "__misc",
        ]


def test_col_as_attribute_class_mangling_25129() -> None:
    # note: we have to run this test in a subprocess to prevent pytest
    # itself from managing to inject "_pytestfixturefunction" as the
    # col name/attribute, where we can't recover the original name
    import subprocess
    import sys

    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
from sys import version_info
import polars as pl
df = pl.DataFrame({"__foo": [0]})

class Mangler:
    def __init__(self):
        self._selected = df.select(pl.col.__foo)

    def foo(self):
        return df.select(pl.col.__foo)

    @classmethod
    def misc(cls):
        def _nested():
            return df.select(pl.col.__foo)
        return _nested()

    @staticmethod
    def indirect():
        return Mangler.misc()

    @staticmethod
    def testing1234():
        return df.select(pl.col.__foo)


# detect mangling in init/instancemethod
assert Mangler()._selected.columns == ["__foo"]
assert Mangler().foo().columns == ["__foo"]

# additionally detect mangling in classmethod/staticmethod
if version_info >= (3, 11):
    assert Mangler.misc().columns == ["__foo"]
    assert Mangler.indirect().columns == ["__foo"]
    assert Mangler.testing1234().columns == ["__foo"]

print("OK", end="")
""",
        ],
    )
    assert out == b"OK"


def test_col_select() -> None:
    df = pl.DataFrame(
        {
            "ham": [1, 2, 3],
            "hamburger": [11, 22, 33],
            "foo": [3, 2, 1],
            "bar": ["a", "b", "c"],
        }
    )

    # Single column
    assert df.select(pl.col("foo")).columns == ["foo"]

    # Regex
    assert df.select(pl.col("*")).columns == ["ham", "hamburger", "foo", "bar"]
    assert df.select(pl.col("^ham.*$")).columns == ["ham", "hamburger"]
    assert df.select(pl.col("*").exclude("ham")).columns == ["hamburger", "foo", "bar"]

    # Multiple inputs
    assert df.select(pl.col(["hamburger", "foo"])).columns == ["hamburger", "foo"]
    assert df.select(pl.col("hamburger", "foo")).columns == ["hamburger", "foo"]
    assert df.select(pl.col(pl.Series(["ham", "foo"]))).columns == ["ham", "foo"]

    # Dtypes
    assert df.select(pl.col(pl.String)).columns == ["bar"]
    for dtype_col in (
        pl.col(NUMERIC_DTYPES),
        pl.col(pl.Int64, pl.Float64),
    ):
        assert df.select(dtype_col).columns == ["ham", "hamburger", "foo"]


def test_col_series_selection() -> None:
    ldf = pl.LazyFrame({"a": [1], "b": [1], "c": [1]})
    srs = pl.Series(["b", "c"])

    assert ldf.select(pl.col(srs)).collect_schema().names() == ["b", "c"]
