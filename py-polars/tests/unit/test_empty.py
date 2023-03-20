import polars as pl
from polars.testing import assert_frame_equal


def test_empty_str_concat_lit() -> None:
    df = pl.DataFrame({"a": [], "b": []}, schema=[("a", pl.Utf8), ("b", pl.Utf8)])
    assert df.with_columns(pl.lit("asd") + pl.col("a")).schema == {
        "a": pl.Utf8,
        "b": pl.Utf8,
        "literal": pl.Utf8,
    }


def test_top_k_empty() -> None:
    df = pl.DataFrame({"test": []})

    assert_frame_equal(df.select([pl.col("test").top_k(2)]), df)


def test_empty_cross_join() -> None:
    a = pl.LazyFrame(schema={"a": pl.Int32})
    b = pl.LazyFrame(schema={"b": pl.Int32})

    assert (a.join(b, how="cross").collect()).schema == {"a": pl.Int32, "b": pl.Int32}


def test_empty_string_replace() -> None:
    s = pl.Series("", [], dtype=pl.Utf8)
    assert s.str.replace("a", "b", literal=True).series_equal(s)
    assert s.str.replace("a", "b").series_equal(s)
    assert s.str.replace("ab", "b", literal=True).series_equal(s)
    assert s.str.replace("ab", "b").series_equal(s)
