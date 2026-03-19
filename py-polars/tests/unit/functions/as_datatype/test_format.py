import polars as pl


def test_format() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt")])
    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]
