import polars as pl
from polars.testing import assert_frame_equal


def test_unique_id_basic() -> None:
    """Test default mode (dense=False): same values get same ID."""
    df = pl.DataFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    result = df.select(pl.col("x").unique_id())
    ids = result["x"].to_list()

    assert result["x"].dtype == pl.UInt32
    # Same values should have same IDs
    assert ids[0] == ids[2]  # A == A
    assert ids[1] == ids[5]  # B == B
    assert ids[3] == ids[4]  # C == C
    # Different values should have different IDs
    assert len(set(ids)) == 3


def test_unique_id_dense() -> None:
    """Test dense=True: IDs are 0..n_unique-1."""
    df = pl.DataFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    result = df.select(pl.col("x").unique_id(dense=True))

    # IDs should be in range 0..2
    assert set(result["x"].to_list()) == {0, 1, 2}


def test_unique_id_dense_maintain_order() -> None:
    """Test dense=True with maintain_order=True: IDs in first-occurrence order."""
    df = pl.DataFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    result = df.select(pl.col("x").unique_id(dense=True, maintain_order=True))

    # A=0, B=1, C=2 (first-occurrence order)
    assert result["x"].to_list() == [0, 1, 0, 2, 2, 1]


def test_unique_id_with_nulls() -> None:
    df = pl.DataFrame({"x": ["A", None, "A", None, "B"]})
    result = df.with_columns(uid=pl.col("x").unique_id())

    # 3 unique values: "A", None, "B"
    assert result["uid"].n_unique() == 3
    # Nulls should get the same ID
    null_ids = result.filter(pl.col("x").is_null())["uid"]
    assert null_ids.n_unique() == 1


def test_unique_id_empty() -> None:
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
    result = df.select(pl.col("x").unique_id())
    expected = pl.DataFrame({"x": pl.Series([], dtype=pl.UInt32)})
    assert_frame_equal(result, expected)


def test_unique_id_single_value() -> None:
    df = pl.DataFrame({"x": ["A", "A", "A"]})
    result = df.select(pl.col("x").unique_id())
    ids = result["x"].to_list()

    # All same values should have same ID
    assert ids[0] == ids[1] == ids[2]


def test_unique_id_group_by() -> None:
    df = pl.DataFrame({"group": ["a", "a", "b", "b"], "x": [1, 2, 1, 2]})
    result = df.group_by("group", maintain_order=True).agg(pl.col("x").unique_id())

    # Each group has 2 different values
    for ids in result["x"].to_list():
        assert len(set(ids)) == 2


def test_unique_id_streaming() -> None:
    """Test streaming with dense=False (parallel processing)."""
    lf = pl.LazyFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    result = lf.select(pl.col("x").unique_id()).collect(engine="streaming")

    # Same values should have same IDs
    ids = result["x"].to_list()
    assert ids[0] == ids[2]  # A == A
    assert ids[1] == ids[5]  # B == B
    assert ids[3] == ids[4]  # C == C


def test_unique_id_streaming_dense() -> None:
    """Test streaming with dense=True (serial processing)."""
    lf = pl.LazyFrame({"x": ["A", "B", "A", "C", "C", "B"]})
    result = lf.select(pl.col("x").unique_id(dense=True, maintain_order=True)).collect(
        engine="streaming"
    )

    assert result["x"].to_list() == [0, 1, 0, 2, 2, 1]
