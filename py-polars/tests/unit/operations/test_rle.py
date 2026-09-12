import polars as pl
from polars.testing.asserts.frame import assert_frame_equal


def test_rle() -> None:
    values = [1, 1, 2, 1, None, 1, 3, 3]
    lf = pl.LazyFrame({"a": values})

    expected = pl.LazyFrame(
        {"len": [2, 1, 1, 1, 1, 2], "value": [1, 2, 1, None, 1, 3]},
        schema_overrides={"len": pl.get_index_type()},
    )

    result_expr = lf.select(pl.col("a").rle()).unnest("a")
    assert_frame_equal(result_expr, expected)

    result_series = lf.collect().to_series().rle().struct.unnest()
    assert_frame_equal(result_series, expected.collect())


def test_rle_id() -> None:
    values = [1, 1, 2, 1, None, 1, 3, 3]
    lf = pl.LazyFrame({"a": values})

    expected = pl.LazyFrame(
        {"a": [0, 0, 1, 2, 3, 4, 5, 5]}, schema={"a": pl.get_index_type()}
    )

    result_expr = lf.select(pl.col("a").rle_id())
    assert_frame_equal(result_expr, expected)

    result_series = lf.collect().to_series().rle_id()
    assert_frame_equal(result_series.to_frame(), expected.collect())


def test_empty_rle_21787() -> None:
    assert pl.Series("a", [], pl.Int64).rle().is_empty()
    assert pl.Series("a", [], pl.Int64).rle_id().is_empty()


def test_rle_over_a_column_of_several_chunks() -> None:
    # A morsel of the streaming engine is a slice of the frame it comes from, which
    # may cross a chunk boundary of it: the values the runs are gathered from are
    # read off a single chunk.
    values = [i % 3 for i in range(33)]
    s = pl.Series("a", values)
    chunked = pl.concat([s.slice(0, 1), s.slice(1, 32)], rechunk=False)
    assert chunked.n_chunks() == 2

    lf = pl.DataFrame([chunked]).lazy()
    expected = pl.DataFrame([s]).lazy().select(pl.col("a").rle())
    for engine in ("in-memory", "streaming"):
        assert_frame_equal(
            lf.select(pl.col("a").rle()).collect(engine=engine),  # type: ignore[arg-type]
            expected.collect(engine=engine),  # type: ignore[arg-type]
        )


def test_rle_over_a_chunk_that_repeats_one_element() -> None:
    # The answer comes off the one element the chunk repeats, rather than off a
    # walk of every element of it: one run, as long as the column.
    for dtype, value in [
        (pl.Int64, 5),
        (pl.String, "ab"),
        (pl.Boolean, True),
        (pl.List(pl.Int64), [1, 2]),
        (pl.Struct({"x": pl.Int64}), {"x": 1}),
        (pl.Datetime("us"), None),
    ]:
        s = pl.select(pl.repeat(pl.lit(value, dtype=dtype), 8).alias("a")).to_series()
        df = pl.DataFrame([s])

        expected = pl.DataFrame(
            {"len": [8], "value": pl.Series("value", [value], dtype=dtype)},
            schema_overrides={"len": pl.get_index_type()},
        )
        assert_frame_equal(df.select(pl.col("a").rle()).unnest("a"), expected)
        assert df.select(pl.col("a").rle_id())["a"].to_list() == [0] * 8
