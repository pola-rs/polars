import polars as pl
from polars.testing import assert_frame_equal


def test_nested_min_max() -> None:
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

    result = df.with_columns(
        pl.max_horizontal(
            pl.min_horizontal("a", "b"), pl.min_horizontal("c", "d")
        ).alias("t")
    )

    expected = pl.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "t": [3]})
    assert_frame_equal(result, expected)
