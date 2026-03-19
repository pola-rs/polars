import polars as pl
from polars.testing import assert_frame_equal


def test_reverse_list_22829() -> None:
    schema = {"a": pl.Int64(), "b": pl.List(pl.Binary)}
    df = pl.DataFrame([], schema=schema, schema_overrides=schema)
    assert df.schema == df.reverse().schema


def test_reverse_df() -> None:
    out = pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).reverse()
    expected = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    assert_frame_equal(out.collect(), expected)


def test_reverse_series() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5])
    assert s.reverse().to_list() == [5, 4, 3, 2, 1]

    s = pl.Series("values", ["a", "b", None, "y", "x"])
    assert s.reverse().to_list() == ["x", "y", None, "b", "a"]


def test_reverse_binary() -> None:
    # single chunk
    s = pl.Series("values", ["a", "b", "c", "d"]).cast(pl.Binary)
    assert s.reverse().to_list() == [b"d", b"c", b"b", b"a"]

    # multiple chunks
    chunk1 = pl.Series("values", ["a", "b"])
    chunk2 = pl.Series("values", ["c", "d"])
    s = chunk1.extend(chunk2).cast(pl.Binary)
    assert s.n_chunks() == 2
    assert s.reverse().to_list() == [b"d", b"c", b"b", b"a"]
