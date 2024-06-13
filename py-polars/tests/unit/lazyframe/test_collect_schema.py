import polars as pl


def test_collect_schema() -> None:
    lf = pl.LazyFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    result = lf.collect_schema()
    expected = pl.Schema({"foo": pl.Int64(), "bar": pl.Float64(), "ham": pl.String()})
    assert result == expected
