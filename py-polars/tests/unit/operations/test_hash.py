import polars as pl
from polars.testing import assert_frame_equal


def test_hash_struct() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.select(pl.struct(pl.all()))
    assert df.select(pl.col("a").hash())["a"].to_list() == [
        5535262844797696299,
        15139341575481673729,
        12593759486533989774,
    ]


def test_hash_cat_stable() -> None:
    c1 = pl.Categories.random()
    c2 = pl.Categories.random()

    # Different insertion order.
    s1 = pl.Series(["cow", "cat", "moo"], dtype=pl.Categorical(c1))
    s2 = pl.Series(["cat", "moo", "cow"], dtype=pl.Categorical(c2))

    # Same data should have same hash.
    df1 = pl.DataFrame(
        {"cat": ["cow", "cat", "moo"]}, schema={"cat": pl.Categorical(c1)}
    )
    df2 = pl.DataFrame(
        {"cat": ["cow", "cat", "moo"]}, schema={"cat": pl.Categorical(c2)}
    )
    assert_frame_equal(
        df1.select(pl.col.cat.hash()),
        df2.select(pl.col.cat.hash()),
    )

    # Also stable in struct?
    df1_struct = df1.select(struct=pl.struct(c=pl.col.cat, x=1))
    df2_struct = df2.select(struct=pl.struct(c=pl.col.cat, x=1))
    assert_frame_equal(
        df1_struct.select(pl.col.struct.hash()),
        df2_struct.select(pl.col.struct.hash()),
    )
