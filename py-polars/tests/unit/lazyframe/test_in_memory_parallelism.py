import polars as pl
from polars.testing import assert_frame_equal


def test_single_chunk_vertical_parallelism_28593() -> None:
    df = pl.DataFrame({"a": range(100_000)})
    assert df.n_chunks() == 1

    predicate = (pl.col("a") * 3 % 7) == 0
    expression = ((pl.col("a") * 3 + 1) % 7).alias("x")

    assert_frame_equal(
        df.lazy().filter(predicate).collect(engine="in-memory"),
        df.filter(predicate),
    )
    assert_frame_equal(
        df.lazy().select(expression).collect(engine="in-memory"),
        df.select(expression),
    )
    assert_frame_equal(
        df.lazy().with_columns(expression).collect(engine="in-memory"),
        df.with_columns(expression),
    )
