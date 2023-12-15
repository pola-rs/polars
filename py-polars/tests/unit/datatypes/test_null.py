import polars as pl
from polars.testing import assert_frame_equal


def test_null_index() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4], [5, 6]], "b": [[1, 2], [1, 2], [4, 5]]})

    result = df.with_columns(pl.lit(None).alias("null_col"))[-1]

    expected = pl.DataFrame(
        {"a": [[5, 6]], "b": [[4, 5]], "null_col": [None]},
        schema_overrides={"null_col": pl.Null},
    )
    assert_frame_equal(result, expected)


def test_null_grouping_12950() -> None:
    assert pl.DataFrame({"x": None}).unique().to_dict(as_series=False) == {"x": [None]}
    assert pl.DataFrame({"x": [None, None]}).unique().to_dict(as_series=False) == {
        "x": [None]
    }
    assert pl.DataFrame({"x": None}).slice(0, 0).unique().to_dict(as_series=False) == {
        "x": []
    }
