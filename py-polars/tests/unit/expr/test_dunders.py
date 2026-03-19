import polars as pl
from polars.testing.asserts.frame import assert_frame_equal


def test_add_parse_str_input_as_literal() -> None:
    df = pl.DataFrame({"a": ["x", "y"]})
    result = df.select(pl.col("a") + "b")
    expected = pl.DataFrame({"a": ["xb", "yb"]})
    assert_frame_equal(result, expected)


def test_truediv_parse_str_input_as_col_name() -> None:
    df = pl.DataFrame({"a": [10, 12], "b": [5, 4]})
    result = df.select(pl.col("a") / "b")
    expected = pl.DataFrame({"a": [2, 3]}, schema={"a": pl.Float64})
    assert_frame_equal(result, expected)
