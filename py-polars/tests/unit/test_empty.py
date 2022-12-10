import polars as pl


def test_empty_str_concat_lit() -> None:
    df = pl.DataFrame({"a": [], "b": []}, columns=[("a", pl.Utf8), ("b", pl.Utf8)])
    assert df.with_column(pl.lit("asd") + pl.col("a")).schema == {
        "a": pl.Utf8,
        "b": pl.Utf8,
        "literal": pl.Utf8,
    }
