import polars as pl


def test_format() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt")])
    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]


def test_format_empty_chunks_25525() -> None:
    s = pl.concat(
        [
            pl.Series("a", ["p"]),
            pl.Series("a", ["x", "y"]),
            pl.Series("a", ["q"]),
            pl.Series("a", ["z", None]),
        ],
        rechunk=False,
    )
    df = pl.DataFrame([s, pl.Series("k", [0, 1, 1, 0, 1, 1])])
    df = df.filter(df["k"] == 1).with_columns(b=pl.Series(["1", "2", "3", "4"]))

    out = df.select(pl.format("<{}>", pl.col("a")))
    assert out.to_series().to_list() == ["<x>", "<y>", "<z>", None]

    out = df.select(pl.format("<{}-{}-{}>", pl.col("b"), pl.lit("s"), pl.col("a")))
    assert out.to_series().to_list() == ["<1-s-x>", "<2-s-y>", "<3-s-z>", None]
